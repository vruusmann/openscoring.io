---
layout: post
title: "Troubleshooting PMML"
author: vruusmann
keywords: jpmml-model testing
---

**CHALLENGE**: you've composed a Predictive Model Markup Language (PMML) document and would like to make sure that it complies with the PMML standard.

**SOLUTION**: perform the following sequence of checks on the PMML document:

1. Is the content structurally valid XML?
2. Is the content structurally valid PMML?
3. Is the content logically valid PMML?

## Structural validation as XML ##

Open the PMML document for inspection in an XML-aware text editor. Most text editors should be able to detect XML content and switch to XML editing mode. 

*Tip*: If the filename extension of the file is not ".xml", consider appending ".xml" to it (at least temporarily) in order to activate OS file type association.

Typically, a PMML document starts with an XML declaration, followed by a `PMML` element:

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
  <!-- Omitted PMML content -->
</PMML>
```

*Tip*: If the content looks messy, consider re-indenting or re-formatting it.

The [JPMML-Model](https://github.com/jpmml/jpmml-model) library provides an `org.jpmml.model.CopyExample` command-line application, which reads/parses a PMML file into an in-memory class model object, and then formats/writes it to another PMML file:

```
$ java -cp pmml-model-example/target/example-1.4-SNAPSHOT.jar org.jpmml.model.CopyExample --input model.pmml.xml --output /dev/null
```

The copy operation fails is the content is not structurally valid XML. The associated error message(s) should convey adequate information about the nature and location of the problem so that it can be manually corrected.

A copy error caused by an incorrect XML namespace URI:

```
Exception in thread "main" java.lang.IllegalArgumentException: PMML namespace URI http://www.dmg.org/v4-2-1 does not match 'http://www\.dmg\.org/PMML\-\d_\d' regex pattern
  at org.dmg.pmml.Version.forNamespaceURI(Version.java:67)
  at org.jpmml.model.filters.PMMLFilter.updateSource(PMMLFilter.java:121)
  at org.jpmml.model.filters.PMMLFilter.startPrefixMapping(PMMLFilter.java:43)
  at com.sun.org.apache.xerces.internal.parsers.AbstractSAXParser.startNamespaceMapping(AbstractSAXParser.java:2164)
  at com.sun.org.apache.xerces.internal.parsers.AbstractSAXParser.startElement(AbstractSAXParser.java:469)
```

A copy error caused by unbalanced XML tags:

```
Caused by: org.xml.sax.SAXParseException; lineNumber: 248; columnNumber: 19; The element type "CompoundPredicate" must be terminated by the matching end-tag "</CompoundPredicate>".
  at com.sun.org.apache.xerces.internal.util.ErrorHandlerWrapper.createSAXParseException(ErrorHandlerWrapper.java:203)
  at com.sun.org.apache.xerces.internal.util.ErrorHandlerWrapper.fatalError(ErrorHandlerWrapper.java:177)
  at com.sun.org.apache.xerces.internal.impl.XMLErrorReporter.reportError(XMLErrorReporter.java:400)
  at com.sun.org.apache.xerces.internal.impl.XMLErrorReporter.reportError(XMLErrorReporter.java:327)
  at com.sun.org.apache.xerces.internal.impl.XMLScanner.reportFatalError(XMLScanner.java:1472)
```

## Structural validation as PMML ##

The JPMML-Model library provides an `org.jpmml.model.ValidationExample` command-line application, which validates the content of a PMML document against the built-in PMML schema definition (XSD) file. This XSD file is based on the latest PMML schema version (at the time of writing this, 4.3), and includes a limited number JPMML vendor extension elements and attributes. Nevertheless, it is suitable for validating all PMML schema version 3.X and 4.X documents, as the PMML standard is fully backwards- and forwards-compatible in this schema version range.

```
$ java -cp pmml-model-example/target/example-1.4-SNAPSHOT.jar org.jpmml.model.ValidationExample --input model.pmml.xml
```

This application prints a list of warnings and errors. Getting to productivity may require some more time and effort, because the printed messages are fairly technical.

Troubleshooting procedure:

1. Open the PMML document in a text editor, and scroll to the pinpointed location.
2. Identify the parent construct of the offending PMML markup; if the parent construct is nested inside some model element, then identify it as well.
3. Open the correct version of PMML specification, and go to the page that deals with the identified parent construct.
4. Jump the (sub-)section that corresponds to the offending PMML markup, figure out the nature of the problem and an appropriate fix. Implement the corrective change manually.
5. Re-run validation to see if the fix was good.

Examples of warnings and errors, and their fixes.

### Missing elements and attributes

Validation log:

```
SEVERE: [severity=FATAL_ERROR,message=cvc-complex-type.2.4.a: Invalid content was found starting with element 'DataDictionary'. One of '{"http://www.dmg.org/PMML-4_3":Header}' is expected.,locator=[node=null,object=null,url=null,line=3,col=41,offset=-1]]
```

The error happens on line 3 of the PMML document, and is about a missing `Header` element. When opening the PMML document in a text editor, then the following content is found:

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" version="4.2">
  <DataDictionary>
    <!-- Omitted field declarations -->
  </DataDictionary>
  <!-- Omitted model -->
</PMML>
```

The parent construct is the `PMML` element. According to the [general structure](https://dmg.org/pmml/v4-4-1/GeneralStructure.html), the first child element of the `PMML` element must be a [`Header`](https://dmg.org/pmml/v4-4-1/Header.html#xsdElement_Header) element, which provides a general description of the model.

The easiest way to satisfy the above requirement is inserting a dummy `Header` element:

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" version="4.2">
  <Header>
    <Application/>
  </Header>
  <DataDictionary>
    <!-- Omitted field declarations -->
  </DataDictionary>
  <!-- Omitted model -->
</PMML>
```

When re-running the validation, then the original error has disappeared, but a new one has shown up:

```
SEVERE: [severity=FATAL_ERROR,message=cvc-complex-type.4: Attribute 'name' must appear on element 'Application'.,locator=[node=null,object=null,url=null,line=4,col=17,offset=-1]]
```

By revisiting the formal specification of the `Application` element, it is reminded that the `name` attribute is required, whereas the `version` attribute is optional:

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_2" version="4.2">
  <Header>
    <Application name="MyConverter" version="0.9-SNAPSHOT"/>
  </Header>
  <DataDictionary>
    <!-- Omitted field declarations -->
  </DataDictionary>
  <!-- Omitted model -->
</PMML>
```

### Invalid and/or misplaced elements and attributes

Validation log:

```
SEVERE: [severity=FATAL_ERROR,message=cvc-complex-type.2.4.a: Invalid content was found starting with element 'Characteristic'. One of '{"http://www.dmg.org/PMML-4_3":ComplexPartialScore}' is expected.,locator=[node=null,object=null,url=null,line=211,col=40,offset=-1]]

SEVERE: [severity=ERROR,message=unexpected element (uri:"http://www.dmg.org/PMML-4_3", local:"Characteristic"). Expected elements are <{http://www.dmg.org/PMML-4_3}Extension>,<{http://www.dmg.org/PMML-4_3}ComplexPartialScore>,<{http://www.dmg.org/PMML-4_3}SimplePredicate>,<{http://www.dmg.org/PMML-4_3}SimpleSetPredicate>,<{http://www.dmg.org/PMML-4_3}CompoundPredicate>,<{http://www.dmg.org/PMML-4_3}False>,<{http://www.dmg.org/PMML-4_3}True>,locator=[node=null,object=null,url=null,line=211,col=40,offset=-1]]
```

These two errors are both reported against the same location (line 211, column 40), which suggests that they are related/indicative of the same problem.

The parent construct is the [`Characteristic`](https://dmg.org/pmml/v4-4-1/Scorecard.html#xsdElement_Characteristic) element, which belongs to the [`Scorecard`](https://dmg.org/pmml/v4-4-1/Scorecard.html#xsdElement_Scorecard) element.

According to the PMML specification, the content of a scorecard model is represented by a sequence of `Characteristic` elements. However, when opening the PMML document in a text editor, then it is possible to find that around line 211 there is a `Characteristic` element nested inside an `Attribute` element (of another `Characteristic` element), which is not permitted:

``` xml
<Characteristics>
  <Characteristic name="X4">
    <Attribute>
      <CompoundPredicate/>
      <Characteristic name="X12">
        <!-- Omitted content -->
      </Characteristic>
    </Attribute>
  </Characteristic>
</Characteristics>
```

Apparently, the unclosed `CompoundPredicate` element that was fixed during the "validation as XML" stage was indicative of a major structural issue - most probably a manual copy and paste rearrangement inside a PMML document (or across a set of PMML documents), where a `Characteristic` element had been pasted into a bad place.

The fix is to relocate the offending `Characteristic` element:

``` xml
<Characteristics>
  <Characteristic name="X4">
    <Attribute>
      <CompoundPredicate/>
    </Attribute>
  </Characteristic>
  <Characteristic name="X12">
    <!-- Omitted content -->
  </Characteristic>
</Characteristics>
```

## Logical validation as PMML ##

The above two structural validation stages should ensure that a PMML document can be loaded by a PMML engine, but they offer no guarantee that the PMML document can be used for prediction, and more importantly, that the predictions will be correct.

Let us consider field definition, scoping and referencing.

Structural validation ensures that all field definition elements are complete. However, a field definition may still be unusable due to PMML type system violations. For example, a `DataField` element, which contains unparseable and duplicate category values:

``` xml
<DataField name="x1" optype="categorical" dataType="integer">
  <Value value="0.0"/>
  <!-- A floating-point numeric value 0.5 cannot be coerced to integer
value space -->
  <Value value="0.5"/>
  <Value value="1.0"/>
  <Value value="1.0"/>
</DataField>
```

Every PMML engine is free to decide if, when and in what way to fail when asked to use the above field definition. A naive PMML engine might not see any problem(s) with it, and never fail. A more more sophisticated but forgiving PMML engine might successfully evaluate `x1 = 0` and `x1 = 1` data records, and only fail with `x1 = 0.5` data records. However, a strict PMML engine might blacklist this field declaration already when loading the PMML document, and not permit the evaluation of any data records.

Field scoping and referencing violations can be fully found via static PMML code analysis. The JPMML-Model library provides a number of Visitor API implementation classes that can traverse the PMML class model object and perform relevant checks.
