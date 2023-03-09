---
layout: post
title: "JPMML-Model: Configuring JAXB dependency for Java SE versions 8, 9, 10 and 11"
author: vruusmann
keywords: jpmml-model
---

The [JPMML-Model](https://github.com/jpmml/jpmml-model) library provides a Java class model for the Predictive Model Markup Language (PMML) standard.
The core set of JPMML-Model classes have been generated based on the PMML XML Schema Definition (XSD) file using the XJC binding compiler technology. As such, they are heavily dependent on the Java XML Binding (JAXB) runtime.

The JAXB runtime was more or less an integral part of Java SE versions 1.6 thorugh 1.8. However, with the advent of [Java SE 9 module system](https://jcp.org/en/jsr/detail?id=376), the JAXB runtime was isolated to a `java.xml.bind` module, and excluded from the core (ie. active by default) module set.

If a Java SE 9 (or newer) application wants to use JAXB runtime functionality, then it can do one of the following:

* Activate the `java.xml.bind` module using the `--add-modules java.xml.bind` command-line option.
* Leave the `java.xml.bind` module inactive, and add a functionally equivalent set of Java libraries/classes straight to the application classpath.

The second option is seen as cleaner and safer. Meddling with Java/JVM startup options just to please one application is bad style. Furthermore, there remains uncertainty about the JAXB runtime version, and if all transitive dependencies (internal APIs for power users) are available and sufficiently up to date.

The JPMML-Model library declares a compile-time dependency ("provided" Apache Maven scope) on [GlassFish Metro](https://javaee.github.io/metro/) and [EclipseLink MOXy](https://www.eclipse.org/eclipselink/) runtimes.

If a Java application declares a run-time dependency ("compile" and "runtime" Apache Maven scopes) only on the JPMML-Model library, then it is limited to the first deployment scenario.
However, if the Java application declares a run-time dependency on the JPMML-Model library **plus** one or more JAXB runtimes, then it can follow either development scenario.

This blog post details the proper configuration of GlassFish Metro runtime. The analysis is based on a `jaxb_demo` demo application, which deals with marshalling and unmarshalling an empty PMML class model object.

The project is built and deployed throughout this exercise using the following sequence of commands:

```
$ export JAVA_HOME=/path/to/jdk
$ mvn clean install
$ $JAVA_HOME/bin/java -cp target/jaxb_demo-executable-1.0-SNAPSHOT.jar jaxb_demo.MarshalDemo > /tmp/jaxb_demo.pmml
$ $JAVA_HOME/bin/java -cp target/jaxb_demo-executable-1.0-SNAPSHOT.jar jaxb_demo.UnmarshalDemo < /tmp/jaxb_demo.pmml
```

### Java SE 1.8(.0_162)

The project can be built and deployed without any formal GlassFish Metro dependency.

The default JAXB runtime does not collect and propagate SAX Locator information, which means that the `org.dmg.pmml.PMMLObject#locator` field is left uninitialized. If the Java application (eg. the unmarshalling demo application) is interested in this information, then it needs to declare the `org.glassfish.jaxb:jaxb-runtime` library as a run-time dependency:

``` xml
<dependency>
  <groupId>org.glassfish.jaxb</groupId>
  <artifactId>jaxb-runtime</artifactId>
  <version>2.3.2</version>
</dependency>
```

### Java SE 9(.0.4) and 10(.0.2)

Apache Maven activates all Java EE modules (including the `java.xml.bind` module) during compilation, so the project can again be built without any formal GlassFish Metro dependency.

However, now and in the future, the `org.glassfish.jaxb:jaxb-runtime` library has become a required run-time dependency. If missing, then any attempt to make use of some JAXB class or interface shall fail with the following `java.lang.ClassNotFoundException`:

```
Exception in thread "main" java.lang.NoClassDefFoundError: javax/xml/bind/JAXBContext
  at jaxb_demo.MarshalDemo.main(MarshalDemo.java:16)
Caused by: java.lang.ClassNotFoundException: javax.xml.bind.JAXBContext
  at java.base/jdk.internal.loader.BuiltinClassLoader.loadClass(BuiltinClassLoader.java:582)
  at java.base/jdk.internal.loader.ClassLoaders$AppClassLoader.loadClass(ClassLoaders.java:185)
  at java.base/java.lang.ClassLoader.loadClass(ClassLoader.java:496)
  ... 1 more
```

The `org.glassfish.jaxb:jaxb-runtime` library currently depends on six other Java libraries:

```
$ mvn dependency:tree

[INFO] --- maven-dependency-plugin:2.8:tree (default-cli) @ jaxb_demo ---
[INFO] jaxb_demo:jaxb_demo:jar:1.0-SNAPSHOT
[INFO] +- org.jpmml:pmml-model:jar:1.4.8:compile
[INFO] \- org.glassfish.jaxb:jaxb-runtime:jar:2.3.2:compile
[INFO]    +- jakarta.xml.bind:jakarta.xml.bind-api:jar:2.3.2:compile
[INFO]    +- org.glassfish.jaxb:txw2:jar:2.3.2:compile
[INFO]    +- com.sun.istack:istack-commons-runtime:jar:3.0.8:compile
[INFO]    +- org.jvnet.staxex:stax-ex:jar:1.8.1:compile
[INFO]    +- com.sun.xml.fastinfoset:FastInfoset:jar:1.2.16:compile
[INFO]    \- jakarta.activation:jakarta.activation-api:jar:1.2.1:compile
```

Library descriptions:

* `org.glassfish.jaxb:jaxb-runtime` - Glassfish Metro runtime. Provides `com.sun.xml.bind.*` classes. Required.
* `com.sun.istack:istack-commons-runtime` - Istack Common Utility Code Runtime. Provides `com.sun.istack.*` classes. Required.
* `com.sun.xml.fastinfoset:FastInfoset` - Fast Infoset Standard for Binary XML. Provides `com.sun.xml.fastinfoset.*` and `org.jvnet.fastinfoset.*` classes. Required when working with binary XML documents; not required when working with text-based XML documents.
* `jakarta.activation:jakarta.activation-api` - JavaBeans Activation Framework API. Provides `javax.activation.*` classes. Required.
* `jakarta.xml.bind:jakarta.xml.bind-api` - Java XML Binding (JAXB) API. Provides `javax.xml.bind.*` classes. Required.
* `org.glassfish.jaxb:txw2` - TXW2 Runtime. Provides `com.sun.xml.txw2.*` classes. Not required.
* `org.jvnet.staxex:stax-ex` - Extensions for StAX API. Provides `org.jvnet.staxex.*` classes. Not required.

For example, declaring a minimal GlassFish Metro dependency:

``` xml
<dependency>
  <groupId>org.glassfish.jaxb</groupId>
  <artifactId>jaxb-runtime</artifactId>
  <version>2.3.2</version>
  <exclusions>
    <exclusion>
      <groupId>com.sun.xml.fastinfoset</groupId>
      <artifactId>FastInfoset</artifactId>
    </exclusion>
    <exclusion>
      <groupId>org.glassfish.jaxb</groupId>
      <artifactId>txw2</artifactId>
    </exclusion>
    <exclusion>
      <groupId>org.jvnet.staxex</groupId>
      <artifactId>stax-ex</artifactId>
    </exclusion>
  </exclusions>
</dependency>
```

These three exclusions reduce the size of the `jaxb_demo` uber-JAR file a bit over 400 kB.

### Java SE 11(.0.2)

[Java SE 11 removed all Java EE modules](https://blog.codefx.org/java/java-11-migration-guide/#Removal-Of-Java-EE-Modules), including the `java.xml.bind` module. Since Apache Maven is unable to provide Java XML Binding classes on its own, the project becomes non-buildable:

```
$ mvn clean install

[ERROR] jaxb_demo/src/main/java/jaxb_demo/MarshalDemo.java:[3,22] package javax.xml.bind does not exist
[ERROR] jaxb_demo/src/main/java/jaxb_demo/MarshalDemo.java:[4,22] package javax.xml.bind does not exist
[ERROR] jaxb_demo/src/main/java/jaxb_demo/MarshalDemo.java:[16,17] cannot find symbol
[ERROR] symbol:   class JAXBContext
[ERROR] location: class jaxb_demo.MarshalDemo
[ERROR] jaxb_demo/src/main/java/jaxb_demo/MarshalDemo.java:[16,39] cannot find symbol
[ERROR] symbol:   variable JAXBContext
[ERROR] location: class jaxb_demo.MarshalDemo
[ERROR] jaxb_demo/src/main/java/jaxb_demo/MarshalDemo.java:[18,17] cannot find symbol
[ERROR] symbol:   class Marshaller
[ERROR] location: class jaxb_demo.MarshalDemo
```

After declaring the minimal GlassFish Metro dependency, the project can be built and deployed as before.

### Resources

* Java application: ["jaxb_demo.zip"]({{ "/resources/2019-02-28/jaxb_demo.zip" | absolute_url }})
