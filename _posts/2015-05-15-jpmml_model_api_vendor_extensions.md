---
layout: post
title: "JPMML-Model: Extending PMML with custom XML content"
author: vruusmann
keywords: jpmml-model mathml
---

The XML data format is commonly despised for its complexity and verbosity, especially in comparison with other text-based data formats such as JSON and YAML. However, there are several application areas where its design proves to be an asset instead of a liability.

One of such application areas is extensibility. An XML document can combine different XML document types. Such mixed XML documents can be safely authored and processed using the most common XML tools and APIs.

The PMML document type is a top-level XML document type. It does not limit the number and kind of embedded XML document types. However, it is rather strict about where custom XML content can be attached to.

There are three main attachment points:

* The [`Annotation`](https://dmg.org/pmml/v4-4-1/Header.html#xsdElement_Annotation) element. Document modification history.
* The [`Extension`](https://dmg.org/pmml/v4-4-1/GeneralStructure.html#xsdElement_Extension) element. Element-specific vendor extensions. Vendor extensions are suitable for adding "depth" to the target element. For example, persisting non-standard data and metadata, which could come in handy during various model life-cycle stages.
Vendor extensions should not be critical for the successful use of the PMML document. The behaviour of a PMML engine should not change (at least, materially) if they are filtered out.
* The [`InlineTable`](https://dmg.org/pmml/v4-4-1/Taxonomy.html#xsdElement_InlineTable) element. Free-form data tables.

The [JPMML-Model](https://github.com/jpmml/jpmml-model) library represents attachment points as List-type fields whose element type is `java.lang.Object`. For example, the `Extension#content` field and the corresponding `Extension#getContent()` getter method are defined as follows:

``` java
package org.dmg.pmml;

@XmlRootElement(name = "Extension", namespace = "http://www.dmg.org/PMML-4_2")
public class Extension extends PMMLObject {

  @XmlMixed
  @XmlAnyElement(lax = true)
  private List<Object> content;


  public List<Object> getContent(){
    if(this.content == null){
      this.content = new ArrayList<Object>();
    }
    return this.content;
  }
}
```

The JPMML-Model library provides full support for producing and consuming mixed PMML documents.

Application developers can choose between two API approaches:

* W3C DOM API. Custom XML content are W3C DOM nodes (ie. instances of `org.w3c.dom.Node`). This approach is applicable to all XML document types, but the development and maintenance costs are rather high. For example, application developers must manually take care of managing XML namespace information.
* Java XML Binding (JAXB) API. Custom XML content are JAXB objects. This approach is applicable to XML document types that have a JAXB class model.

This blog post details a method for working with PMML documents that embed MathML content. [Mathematical Markup Language (MathML)](https://en.wikipedia.org/wiki/MathML) is an XML-based standard for describing mathematical notations and capturing both its structure and content. It is potentially useful for adding human- and machine-readable documentation to data transformations.

The XML Schema Definition (XSD) for MathML version 3 is is readily available. It can be compiled to JAXB class model with the help of the XJC binding compiler. The generated MathML class model consists of a number of classes in the `org.wc3.math` package. By convention, the XML registry class is named `org.w3c.math.ObjectFactory`.

### Production

##### W3C DOM approach

The MathML content is captured as an `org.w3c.dom.Element` object. It is critical that the W3C DOM API is operated in an XML namespace-aware fashion. This involves invoking the `javax.xml.parsers.DocumentBuilderFactory#setNamespaceAware(boolean)` method with `true` as an argument and using the `org.w3c.dom.Document#createElementNS(String, String)` method for creating `Element` nodes.

``` java
private static final DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();

static {
  documentBuilderFactory.setNamespaceAware(true);
}

private org.wc3.dom.Element createMathML(){
  DocumentBuilder documentBuilder = documentBuilderFactory.newDocumentBuilder();

  Document document = documentBuilder.newDocument();

  Element element = document.createElementNS("http://www.w3.org/1998/Math/MathML", "mathml:math");
  // Create MathML content
  // All child elements should specify the same namespace URI ("http://www.w3.org/1998/Math/MathML") and namespace prefix ("mathml") as the root element

  return element;
}
```

The completed MathML object is then appended to the list of vendor extensions:

``` java
/**
 * Wraps a MathML object into an {@link Extension} element and appends it to the live list of extensions.
 *
 * @param hasExtensions A PMML object that supports extensions.
 * @param mathObject A MathML W3C DOM node or JAXB object.
 */
static
public void addMathML(HasExtensions hasExtensions, Object mathObject){
  List<Extension> extensions = hasExtensions.getExtensions();

  Extension extension = new Extension()
    .addContent(mathObject);

  extensions.add(extension);
}
```

A PMML class model object that is enriched with custom XML content in the form of W3C DOM nodes can be marshalled to a PMML document using the `org.jpmml.model.JAXBUtil#marshalPMML(org.dmg.pmml.PMML, javax.xml.transform.Result)` utility method.

The resulting mixed content PMML document:

``` xml
<?xml version="1.0" standalone="yes"?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" version="4.2">
  <!-- Omitted PMML content -->
  <Extension>
    <mathml:math xmlns:mathml="http://www.w3.org/1998/Math/MathML">
      <!-- Omitted MathML content -->
    </mathml:math>
  </Extension>
</PMML>
```

##### JAXB approach

The MathML content is captured as a `org.w3c.math.Math` object. The JAXB class model classes are properly annotated with XML namespace information. Therefore, application developers can focus on high-level work with POJOs.

``` java
private org.w3c.math.Math createMathML(){
  Math math = new Math();
  // Create MathML content

  return math;
}
```

Unlike with the W3C DOM approach, a PMML class object that is enriched with JAXB objects cannot be marshalled using the `JAXBUtil#marshalPMML(PMML, Result)` utility method because of the following exception:

```
javax.xml.bind.MarshalException - with linked exception:
[com.sun.istack.internal.SAXException2: class org.w3c.math.Math nor any of its super class is known to this context.
javax.xml.bind.JAXBException: class org.w3c.math.Math nor any of its super class is known to this context.]
  at com.sun.xml.internal.bind.v2.runtime.MarshallerImpl.write(MarshallerImpl.java:311)
  at com.sun.xml.internal.bind.v2.runtime.MarshallerImpl.marshal(MarshallerImpl.java:236)
  ... 3 more
Caused by: com.sun.istack.internal.SAXException2: class org.w3c.math.Math nor any of its super class is known to this context.
javax.xml.bind.JAXBException: class org.w3c.math.Math nor any of its super class is known to this context.
  at com.sun.xml.internal.bind.v2.runtime.XMLSerializer.reportError(XMLSerializer.java:235)
  at com.sun.xml.internal.bind.v2.runtime.XMLSerializer.reportError(XMLSerializer.java:250)
  at com.sun.xml.internal.bind.v2.runtime.property.ArrayReferenceNodeProperty.serializeListBody(ArrayReferenceNodeProperty.java:107)
  ... 28 more
Caused by: javax.xml.bind.JAXBException: class org.w3c.math.Math nor any of its super class is known to this context.
  at com.sun.xml.internal.bind.v2.runtime.JAXBContextImpl.getBeanInfo(JAXBContextImpl.java:564)
  at com.sun.xml.internal.bind.v2.runtime.property.ArrayReferenceNodeProperty.serializeListBody(ArrayReferenceNodeProperty.java:97)
  ... 28 more
```

Just like the exception message suggests, the solution is to make the MathML class model known to the JAXB runtime.

A `javax.xml.bind.JAXBContext` objects can be created by invoking the `JAXBContext#newInstance(Class...)` method with a list of XML registry classes as arguments. Currently, this list must include `org.dmg.pmml.ObjectFactory` and `org.w3c.math.ObjectFactory` classes.

A custom `JAXBContext` object passes complete type information to its child `javax.xml.bind.Marshaller` and `javax.xml.bind.Unmarshaller` objects. The marshalling and unmarshalling behaviour can be further modified by adjusting generic as well as implementation-specific configuration options. For example, setting the generic `jaxb.formatted.output` configuration option to `true` will indentate the XML document to make it more human friendly:

``` java
private static JAXBContext mixedContext = null;

static
private JAXBContext getOrCreateMixedContext() throws JAXBException {

  if(mixedContext == null){
    mixedContext = JAXBContext.newInstance(org.dmg.pmml.ObjectFactory.class, org.w3c.math.ObjectFactory.class);
  }

  return mixedContext;
}

static
public Marshaller createMixedMarshaller() throws JAXBException {
  JAXBContext context = getOrCreateMixedContext();

  Marshaller marshaller = context.createMarshaller();
  marshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, Boolean.TRUE);

  return marshaller;
}

static
public Unmarshaller createMixedUnmarshaller() throws JAXBException {
  JAXBContext context = getOrCreateMixedContext();

  Unmarshaller unmarshaller = context.createUnmarshaller();

  return unmarshaller;
}
```

The resulting mixed content PMML document:

``` xml
<?xml version="1.0" standalone="yes" ?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" xmlns:ns2="http://www.w3.org/1998/Math/MathML" version="4.2">
  <!-- Omitted PMML content -->
  <Extension>
    <ns2:math>
      <!-- Omitted MathML content -->
    </ns2:math>
  </Extension>
</PMML>
```

### Consumption

The "completeness" of unmarshalling operation depends on which JAXB class models are known to the JAXB runtime. In brief, known XML content is returned in the form of JAXB objects, whereas unknown XML content is returned in the form of W3C DOM nodes.

The `JAXBUtil#unmarshalPMML(javax.xml.transform.Source)` utility method is only aware of the PMML class model. Therefore, all custom XML content is returned in the form of W3C DOM nodes. In that sense, `JAXBUtil#marshalPMML(PMML, Result)` and `JAXBUtil#unmarshal(Source)` utility methods are reciprocal to each other.

Sometimes it may be desirable to defer the unmarshalling of custom XML content. For example, the PMML content is unmarshalled completely during the first pass, whereas MathML content is unmarshalled either completely or on an XML node basis (eg. rendering a tooltip) during subsequent passes.

The following code replaces MathML W3C DOM nodes with the corresponding JAXB objects:

``` java
private static JAXBContext mathMlContext = null;

static
private JAXBContext getOrCreateMathMLContext() throws JAXBException {

  if(mathMlContext == null){
    mathMlContext = JAXBContext.newInstance(org.w3c.math.ObjectFactory.class);
  }

  return mathMlContext;
}

/**
 * @param hasExtensions A PMML object that supports extensions.
 */
static
public void unmarshalMathML(HasExtensions hasExtensions) throws JAXBException {
  JAXBContext context = getOrCreateMathMLContext();

  Unmarshaller unmarshaller = context.createUnmarshaller();

  List<Extension> extensions = hasExtensions.getExtensions();
  for(Extension extension : extensions){
    List<Object> content = extension.getContent();

    for(int i = 0; i < content.size(); i++){
      Object object = content.get(i);

      if(object instanceof org.w3c.dom.Element){
        org.w3c.dom.Element element = (org.w3c.dom.Element)object;
        org.w3c.math.Math math = (org.w3c.math.Math)unmarshaller.unmarshal(new DOMSource(element));

        content.set(i, math);
      }
    }
  }
}
```
