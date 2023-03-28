---
layout: post
title: "JPMML-Evaluator: Preparing arguments for evaluation"
author: vruusmann
keywords: jpmml-evaluator
---

The central piece of the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library is the `org.jpmml.evaluator.Evaluator` interface, which declares `#prepare(FieldName, Object)` and `#evaluate(Map<FieldName, ?>)` methods. This API dates back to earliest versions (i.e. 1.0.2) and is still going strong.

This blog post details the relationship between those two methods.

Quite naturally, data preparation precedes data evaluation. It involves three activities:

* Conversion of values from the Java type system to the PMML type system. PMML has a two-tier type system, where the first level relates to data type (enumeration `org.dmg.pmml.DataType`) and the second level relates to operational type (enumeration `org.dmg.pmml.OpType`). For example, a Java string could either become a categorical PMML string or an ordinal PMML string, which exhibit different behavior in comparison operations.
* Validation of values as specified by the `DataField` element.
* Treatment of invalid, outlier and missing values as specified by the `MiningField` element.

The JPMML-Evaluator library represents PMML values using subclasses of the `org.jpmml.evaluator.FieldValue` class (beware, the [JPMML-Model](https://github.com/jpmml/jpmml-model) library contains a class with the same simple name `org.dmg.pmml.FieldValue`). Most model types operate on scalar-type field values. However, there are some model types such as [association rules model](https://dmg.org/pmml/v4-4-1/AssociationRules.html) and [sequence rules model](https://dmg.org/pmml/v4-4-1/Sequence.html) that operate on Collection-type field values. Application developers are advised to employ the `org.jpmml.evaluator.FieldValueUtil` utility class whenever there is a need to create new or refine existing (e.g. change data or operational type) field values.

## Option 1: Eager preparation ##

The classical approach is to create a new argument map, and populate it with prepared field values one by one:

``` java
public Map<FieldName, ?> prepareEagerlyAndEvaluate(Evaluator evaluator, Map<String, ?> userArguments){
  Map<FieldName, FieldValue> pmmlArguments = new LinkedHashMap<FieldName, FieldValue>();

  List<FieldName> activeFields = evaluator.getActiveFields();
  for(FieldName activeField : activeFields){
    // The key type of the user arguments map is java.lang.String.
    // A FieldName can be "unwrapped" to a String using FieldName#getValue().
    Object userValue = userArguments.get(activeField.getValue());

    // The value type of the user arguments map is unknown.
    // An Object is converted to a String using Object#toString().
    // A missing value is represented by null.
    FieldValue pmmlValue = evaluator.prepare(activeField, (userValue != null ? userValue.toString() : null));

    pmmlArguments.put(activeField, pmmlValue);
  }

  return evaluator.evaluate(pmmlArguments);
}
```

This approach is the most versatile one. The `userArguments` variable could be any map-like data structure, including a query interface that fetches data interactively (e.g. prompts the end user). Application developers have full control over handling data preparation errors.

Typically, the `pmmlArguments` variable is serializable using Java's serialization mechanism (i.e. the whole object graph implements the `java.io.Serializable` interface). This allows developing distributed applications where data preparation and data evaluation are separated from each other.

The `Evaluator#prepare(FieldName, Object)` method only deals with scalar-type field values. A Collection-type field value must be subjected to data preparation element-wise. Application developers are advised to employ the `org.jpmml.evaluator.EvaluatorUtil#prepare(Evaluator, FieldName, Object)` utility method when handling a mix of scalar- and Collection-type field values.

## Option 2: Lazy preparation ##

The modern approach is to dispatch user arguments as they are:

``` java
public Map<FieldName, ?> prepareLazilyAndEvaluate(Evaluator evaluator, Map<FieldName, String> userArguments){
  return evaluator.evaluate(userArguments);
}
```

This approach is the most concise one. Essentially, the interaction with the JPMML-Evaluator library is reduced to a single line of code, which greatly simplifies application maintenance. The downside is less control over data preparation errors. The `Evaluator#evaluate(Map<FieldName, ?>)` method fails when the first problematic field value is encountered. In other words, the whole data record is invalidated, not just some field(s).

This approach is fully supported by JPMML-Evaluator version 1.1.4 and newer. Earlier versions implement the conversion of values, but do not implement the validation of values and treatment of invalid, outlier and missing values (see above). Even though the data evaluation operation is very likely to succeed with earlier versions, the result is unspecified in terms of the PMML specification (e.g. may complete successfully instead of failing with a PMML invalid field value exception).

The recommended type for argument map values is `java.lang.String`. A Java string can be parsed into any PMML type provided that it is syntactically and semantically correct. The parsing overhead is negligible. There is no need for "optimizations" such as pre-parsing Java strings to Java primitive values in application code. In fact, doing so may lead to a PMML type cast exception afterwards.

The JPMML-Evaluator library does not make any guarantees exactly when and where the data preparation operation is executed. This should leave room for implementing more sophisticated field value preparation and caching data flows in future versions.

The following Java source code approximates the lazy loading logic inside the `org.jpmml.evaluator.EvaluationContext` class:

``` java
private FieldValue getFieldValue(Evaluator evaluator, Map<FieldName, ?> arguments, FieldName field){
  Object value = arguments.get(field);

  // Return as-is
  if(value instanceof FieldValue){
    return (FieldValue)value;
  }

  return evaluator.prepare(field, value);
}
```

## Option 3: Manual preparation ##

The lazy loading logic provides a "loophole", which makes it possible to circumvent data preparation altogether when `FieldValue` objects are created manually:

``` java
public Map<FieldName, ?> prepareManuallyAndEvaluate(Evaluator evaluator, Map<FieldName, Double> userArguments){
  Map<FieldName, FieldValue> pmmlArguments = new LinkedHashMap<FieldName, FieldValue>();

  List<FieldName> activeFields = evaluator.getActiveFields();
  for(FieldName activeField : activeFields){
    Double userValue = userArguments.get(activeField);

    FieldValue pmmlValue = FieldValueUtil.create(DataType.DOUBLE, OpType.CONTINUOUS, userValue);

    pmmlArguments.put(activeField, pmmlValue);
  }

  return evaluator.evaluate(pmmlArguments);
}
```

This approach assumes that the application code takes full responsibility for data preparation. The replacement of PMML data preparation logic with application code should improve execution speeds (moreover, the majority of PMML converters appear to be generating no-op `DataField` and `MiningField` elements anyway). This approach is relatively more advantageous in situations where the data record contains a large number of fields, which are updated only partially (e.g. ten fields out of one hundred fields) between subsequent runs.
