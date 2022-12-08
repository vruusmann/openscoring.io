---
layout: post
title: "JPMML-Model: Transforming and measuring the memory consumption of class model objects using the Java agent technology"
author: vruusmann
keywords: jpmml-model memory optimization
---

Java (programming language-) agent is a JVM service that is based on the [Java Instrumentation API](https://docs.oracle.com/javase/8/docs/api/java/lang/instrument/package-summary.html). Java agents are loaded into the JVM and activated before any application code is loaded. Therefore, Java agents have the unique ability to monitor and/or control the complete life-cycle of Java applications. This is typically achieved by modifying the definitions of Java class files.

### SAX Locator information

The [JPMML-Model](https://github.com/jpmml/jpmml-model) library provides a class model that is rooted at the `org.dmg.pmml.PMMLObject` class. This class declares a sole `locator` field, whose responsibility is to hold SAX Locator information. Different JAXB runtimes are able to discover and initialize this field in a completely automated fashion, because it is marked with appropriate proprietary annotations (eg. `com.sun.xml.bind.annotation.XmlLocation` for [GlassFish Metro implementation](https://jaxb.java.net/), `org.eclipse.persistence.oxm.annotations.XmlLocation` for [EclipseLink MOXy implementation](https://eclipse.org/eclipselink/moxy.php)).

Application developers can access SAX Locator information using the `org.dmg.pmml.HasLocator` interface.

``` java
public Locator getLocator(Object object){

  if(object instanceof HasLocator){
    HasLocator hasLocator = (HasLocator)object;

    return hasLocator.getLocator();
  }

  return null;
}
```

The `HasLocator#getLocator()` method returns an `org.xml.sax.Locator` object when the PMML document was unmarshalled from a SAX source or a SAX-backed DOM source. It returns a `null` reference when the PMML document was unmarshalled from other types of sources, or created manually.

SAX Locator information is relevant for PMML engines, especially in model development and model testing stages. For example, the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library uses it to point out the exact location of the problematic class model object in the source PMML document when throwing a subclass of `org.jpmml.manager.PMMLException`. However, if the quality assurance process is up to the challenge, then there should be no place for such debugging work in the final model deployment stage.

Conversely, SAX Locator information is absolutely irrelevant for PMML converters. For example, it cannot be used to order the JAXB runtime to format the PMML document in a specific way when marshalling.

This leads to the conclusion that, more often than not, it would be desirable to get rid of the `locator` field in a safe and easy manner. The main benefit of doing so is that it reduces the memory consumption by 25-30%. Given that RAM is cheap and plentiful nowadays, this optimization becomes reasonable to pursue when the application needs to deploy a large number of ensemble models (eg. Random Forest models) in parallel. The added benefit is that the unmarshalling time is reduced in the same proportion.

### Activating the JPMML agent

JPMML agent is part of the JPMML-Model library project. JPMML agent depends on the [Javassist](https://www.javassist.org) library for its Java class file transformation functionality. Both the JPMML agent JAR file and the Javassist JAR file can be downloaded from the Maven Central repository:

* [Search for a JPMML agent JAR file](https://search.maven.org/#search%7Cgav%7C1%7Cg%3A%22org.jpmml%22%20AND%20a%3A%22pmml-agent%22).
* [Search for a Javassist JAR file](https://search.maven.org/#search%7Cgav%7C1%7Cg%3A%22org.javassist%22%20AND%20a%3A%22javassist%22).

The following example assumes that the Java application is packaged into an executable JAR file `myapplication.jar` and the name of the main class is `com.mycompany.myapplication.Main`.

Executing the application:

```
$ java -jar myapplication.jar
```

JPMML agent is loaded into the JVM using the `-javaagent` option. It takes an optional boolean argument `transform`, which indicates if the `locator` field should be deleted (true) or not (false):

Executing the application with the JPMML agent in "non-transforming mode":

```
$ java -javaagent:pmml-agent-1.1.14.jar -jar myapplication.jar
```

Executing the application with the JPMML agent in "transforming mode":

```
$ java -javaagent:pmml-agent-1.1.14.jar=transform=true -cp javassist-3.19.0-GA.jar:myapplication.jar com.mycompany.myapplication.Main
```

If the JAR file `myapplication.jar` does not contain Javassist classes, then they need to be added to the application classpath by other means. The JVM ignores the `-cp` option when the `-jar` option is set. Therefore, in the last command, the application classpath is crafted manually by prepending the Javassist JAR file to the application JAR file, and the name of the main class is spelled out in full.

### Transformation

Transformation is performed by the `org.jpmml.agent.PMMLObjectTransformer` class. The current implementation is naive, because the SAX Locator information is omitted simply by deleting the `locator` field. A more sophisticated implementation could perform a series of "push down" refactorings, so that this field is preserved for subclasses that are associated with more error-prone PMML content.

Java source code representation of the `PMMLObject` class before transformation:

``` java
package org.dmg.pmml;

import java.io.Serializable;

import javax.xml.bind.annotation.XmlTransient;

import org.xml.sax.Locator;

abstract
public class PMMLObject implements HasLocator, Serializable {

  @XmlTransient
  @com.sun.xml.bind.annotation.XmlLocation
  @org.eclipse.persistence.oxm.annotations.XmlLocation
  private Locator locator = null;


  public Locator getLocator(){
    return this.locator;
  }

  public void setLocator(Locator locator){
    this.locator = locator;
  }
}
```

The same after transformation:

``` java
package org.dmg.pmml;

import java.io.Serializable;

import org.xml.sax.Locator;

abstract
public class PMMLObject implements HasLocator, Serializable {

  public Locator getLocator(){
    return null;
  }

  public void setLocator(Locator locator){
  }
}
```

This transformation should be completely safe and undetectable from the Java application perspective.

### Memory measurement

Memory measurement is performed by the `org.jpmml.model.visitors.MemoryMeasurer` class that traverses a class model object first by JPMML-Model Visitor API and then by Java Reflection API. This Visitor maintains a set of distinct objects that are reachable from the specified "root" object. The size of individual objects is approximated using the `java.lang.instrument.Instrumentation#getObjectSize(Object)` method. The total size of the class model object is calculated by summing the sizes of set members.

The decision to implement yet another memory measurement tool (as opposed to reusing some existing tool, eg. [Java Agent for Memory Measurements](https://github.com/jbellis/jamm)) is supported by specific traits of the JPMML-Model class model:

* The traversal by JPMML-Model Visitor API is much faster than by Java Reflection API. Speed becomes critical when working with extremely large (several GB in size) ensemble models.
* Proper handling of PMML class model-specific data types. For example, `org.dmg.pmml.FieldName` objects are treated as `enum` constants when they are interned, and as regular objects when they are not interned.
* Conservative definition of "reachability". For example, interned `java.lang.String` objects and shared Java primitive value wrapper objects are added to the set of distinct objects.

Java source code of a simple application that outputs basic information about a class model object:

``` java
public class Main {

  static
  public void main(String... args) throws Exception {
    PMML pmml;

    InputStream is = new FileInputStream(args[0]);

    try {
      pmml = JAXBUtil.unmarshalPMML(new StreamSource(is));
    } finally {
      is.close();
    }

    MemoryMeasurer measurer = new MemoryMeasurer();
    measurer.applyTo(pmml);

    Set<Object> objects = measurer.getObjects();
    System.out.println("The number of distinct objects in the object graph: " + objects.size());

    long size = measurer.getSize();
    System.out.println("The size of the object graph: " + size + " bytes");
  }
}
```

Memory measurements are performed on the already familiar PMML document `RandomForestIris.pmml`.

The results in "non-transforming mode":

```
$ java -version
java version "1.8.0_31"
Java(TM) SE Runtime Environment (build 1.8.0_31-b13)
Java HotSpot(TM) 64-Bit Server VM (build 25.31-b07, mixed mode)

$ java -javaagent:pmml-model-1.1.14.jar -cp javassist-3.19.0-GA.jar:example.jar Main RandomForestIris.pmml
The number of distinct objects in the object graph: 373
The size of the object graph: 13680 bytes
```

The results in "transforming mode":

```
$ java -javaagent:pmml-model-1.1.14.jar=transform=true -cp javassist-3.19.0-GA.jar:example.jar Main RandomForestIris.pmml
The number of distinct objects in the object graph: 271
The size of the object graph: 9920 bytes
```

All the differences between these two object graphs are solely attributable to the omission of SAX Locator information. It can be seen that `org.xml.sax.Locator` objects make up (373 - 271) / 373 = 27.3% of distinct objects and (13680 - 9920) / 13680 = 27.5% of memory consumption.
