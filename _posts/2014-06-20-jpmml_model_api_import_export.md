---
layout: post
title: "JPMML-Model: Converting PMML between different schema versions"
author: vruusmann
keywords: jpmml-model
---

The [Data Mining Group](http://www.dmg.org) has been working diligently to ensure backward compatibility between PMML schema versions. The PMML specification uses two-level versioning system with the general formula of `<major version>.<minor version>`. The major version number is tied to the overall structure of the PMML document (e.g. data flows, data pre- and post-processing). The minor version number is tied to specific features (e.g. model types). The addition of new features is straightforward. The removal of features is a two-step process, where the feature is first marked as deprecated and then removed once the grace period is over.

The [JPMML-Model](https://github.com/jpmml/jpmml-model) library provides a class model that is currently capable of representing all PMML schema version 3.0, 3.1, 3.2, 4.0, 4.1 and 4.2 documents. On a side note, there are no plans for supporting earlier PMML schema versions (i.e. 1.X and 2.X), because they date back more than a decade and are effectively obsolete.

The class model is generated after an XML Schema Definition (XSD) file. This XSD file is based on the latest PMML schema version 4.2 XSD file, which has been edited to restore all the features that have been removed in PMML schema versions 3.0 through 4.1. The class model is enhanced using the following version annotation classes:

* `org.jpmml.schema.Added`. Marks a feature that has been added in the specified version.
* `org.jpmml.schema.Deprecated` (not to be confused with Java's `java.lang.Deprecated`). Marks a feature that has been deprecated in the specified version.
* `org.jpmml.schema.Removed`. Marks a feature that has been removed in the specified version.

When working with version annotations, then it is worth stressing over that the `@Added` includes the value, whereas `@Deprecated` and `@Removed` exclude it. For example, the class model defines the `ruleFeature` attribute of the `OutputField` element as follows:

``` java
@Added(Version.PMML_4_0)
@Deprecated(Version.PMML_4_2)
protected RuleFeatureType ruleFeature;
```

This declaraton states that the `ruleFeature` attribute was added in PMML schema version 4.0 and deprecated in PMML schema version 4.2. In other words, it is a first-class concept in PMML schema version 4.0 and 4.1 documents. It can be used in PMML schema version 4.2 documents, but doing so is discouraged, because it has been superseded by another set of attributes. In any way, the `ruleFeature` attribute cannot be used in PMML schema version 3.2 and earlier documents. A validating PMML parser would report that as an error.

The JPMML-Model library provides a `org.jpmml.model.SchemaInspector` Visitor class that traverses a class model object and computes its supported version range. The upper and lower boundaries can be queried using `#getMinimum()` and `#getMaximum()` methods, respectively. The following Java source code checks if a class model object is compatible with the specified PMML schema version:

``` java
public boolean isCompatible(PMMLObject object, Version version){
  SchemaInspector inspector = new SchemaInspector();

  // Traverse the class model object
  object.accept(inspector);

  // Detect features that have been added after the target version
  int minDiff = version.compareTo(inspector.getMinimum());
  if(minDiff < 0){
    return false;
  }

  // Detect features that have been removed before the target version
  int maxDiff = version.compareTo(inspector.getMaximum());
  if(maxDiff > 0){
    return false;
  }

  return true;
}
```

The conversion of PMML documents includes the following activities:

* Updating the XML namespace declaration.
* Updating the `version` attribute of the `PMML` element.
* Updating the name of renamed elements and attributes.

These activities can be implemented using XML filtering. More complicated activities (e.g. replacing a deprecated feature with an up-to-date feature) should be handled in Java application code. The JPMML-Model library is expected to provide a collection of such programmatic converters in the future.

The XML filtering allows for direct conversion between arbitrary PMML schema versions. However, it is recommended to employ an intermediated conversion approach, where the input PMML document is first parsed to an in-memory PMML schema version 4.2 class model object, which is validated ("trust, but verify") and only then formatted to the output PMML document.

The conversion from any PMML schema version 3.X or 4.X document to a PMML schema version 4.2 document is implemented by the `org.jpmml.model.ImportFilter` class. This filter should be applied to the source before feeding it to the PMML unmarshaller:

``` java
public PMML readPMML(InputStream is) throws Exception {
  ImportFilter filter = new ImportFilter(XMLReaderFactory.createXMLReader());

  SAXSource filteredSource = new SAXSource(filter, new InputSource(is));

  return JAXBUtil.unmarshalPMML(filteredSource);
}
```

The conversion in the opposite direction is implemented by the `org.jpmml.model.ExportFilter` class. Java's simple API for XML (SAX) does not provide means for applying XML filters to results. In theory, it should be possible to perform XML filtering on a result obtained from the PMML marshaller using a generic XSL identity transformation. In practice, however, it fails to update the XML namespace declaration for an unknown reason. The following Java source code performs a SAX-specific transformation:

``` java
public void writePMML(PMML pmml, Version version, OutputStream os) throws Exception {

  // Avoid producing invalid PMML documents
  if(!isCompatible(pmml, version)){
    throw new IllegalArgumentException("The class model object is not compatible with PMML schema version " + version);
  }

  SAXTransformerFactory transformerFactory = (SAXTransformerFactory)SAXTransformerFactory.newInstance();

  TransformerHandler transformerHandler = transformerFactory.newTransformerHandler();
  transformerHandler.setResult(new StreamResult(os));

  ExportFilter filter = new ExportFilter(XMLReaderFactory.createXMLReader(), version);
  filter.setContentHandler(transformerHandler);

  filter.parse(toInputSource(pmml));
}

private InputSource toInputSource(PMML pmml) throws Exception {
  ByteArrayOutputStream os = new ByteArrayOutputStream();

  JAXBUtil.marshalPMML(pmml, new StreamResult(os));

  ByteArrayInputStream is = new ByteArrayInputStream(os.toByteArray());

  return new InputSource(is);
}
```

**Update**: Starting from JPMML-Model version 1.1.12, the `org.jpmml.model.SchemaInspector` class has been relocated and renamed to the `org.jpmml.model.visitors.VersionInspector` class.
