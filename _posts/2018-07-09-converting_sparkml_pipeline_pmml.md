---
layout: post
title: "Converting Apache Spark ML pipelines to PMML"
author: vruusmann
keywords: apache-spark pyspark jpmml-sparkml pyspark2pmml sparklyr sparklyr2pmml
---

The [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library converts Apache Spark ML pipeline models to the standardized Predictive Model Markup Language (PMML) representation.

The project has been around for more than two years by now.
The first iteration defined public API entry point in the form of the `org.jpmml.sparkml.ConverterUtil` utility class. Subsequent iterations have been mostly about adding support for more transformer and model types, and expanding Apache Spark version coverage.
However, recent iterations have been gradually introducing new public API building blocks, and the last iteration (26 June, 2018) made them official.

This blog post details this breaking API change, and all the new features and functionality that it entails.

## API overview ##

The old API was designed after Apache Spark MLlib's [`org.apache.spark.mllib.pmml.PMMLExportable`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/pmml/PMMLExportable.html) trait.
The `ConverterUtil#toPMML(StructType, PipelineModel)` utility method was simply doing its best to emulates the non-existing `org.apache.spark.ml.PipelineModel#toPMML(StructType)` method.

The new API is designed after the [builder pattern](https://en.wikipedia.org/wiki/Builder_pattern).
The primary (ie. essential) state of the `org.jpmml.sparkml.PMMLBuilder` class includes the dataset schema and the fitted pipeline. The initial values are supplied via the the two-argument `PMMLBuilder(StructType, PipelineModel)` constructor, and can be updated any time via the `#setSchema(StructType)` and `#setPipelineModel(PipelineModel)` mutator methods.

Schema mutations may involve renaming columns or clarifying their data types.
Apache Spark ML pays little attention to the data type of categorical features, because after string indexing, vector indexing, vector slicing and dicing, they all end up as `double` arrays anyway.
In contrast, JPMML-SparkML carefully collects and maintains (meta-)information about each and every feature, with the aim of using it to generate more precise and nuanced PMML documents. For example, JPMML-SparkML (just like all other JPMML conversion libraries) eagerly takes note if the data type of a column is indicated as `boolean`, and generates simplified, binary logic PMML language constructs wherever possible.

Pipeline mutation may involve inserting new transformers and models, or removing existing ones. For example, generating a "full pipeline" PMML document, then removing all transformers and generating a "model-only" PMML document.

The secondary (ie. non-essential) state of the `PMMLBuilder` class includes conversion options and verification data.

Application code should treat `PMMLBuilder` objects as local throwaway objects.
Due to the tight coupling to the Apache Spark environment, they are not suitable for persistence, or exchanging between applications and environments.

## Choosing the right JPMML-SparkML flavour and version ##

JPMML-SparkML exists in two flavours:

* Library JAR file `jpmml-sparkml-${version}.jar`. Contains `org.jpmml.sparkml.*` classes. Distributed via the Maven Central repository.
* Executable uber-JAR file `jpmml-sparkml-executable-${version}.jar`. Contains all library JAR file classes, plus all transitive dependency (JPMML-Converter, JPMML-Model, Google Guava, etc.) classes. Distributed via the [GitHub releases](https://github.com/jpmml/jpmml-sparkml/releases) page.

The "business logic" of Apache Spark ML transformers and models is version dependent. Major releases (eg. `2.<major>`) introduce new algorithms and parameterization schemes, whereas minor versions (eg. `2.<major>.<minor>`) address their stability and optimization issues.

JPMML-SparkML is versioned after Apache Spark major versions. There is an active JPMML-SparkML development branch for every supported Apache Spark major version. The conversion logic for some transformer or model is implemented in the earliest development branch, and then merged forward to later development branches (all while accumulating customizations and changes)

At the time of writing this (July 2018; updated in January 2019), JPMML-SparkML supports all current Apache Spark 2.X versions:

| Apache Spark version | JPMML-SparkML development branch | JPMML-SparkML latest release version |
|----------------------|----------------------------------|--------------------------------------|
| 2.0.X | [`1.1.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.1.X) | [`1.1.23`](https://github.com/jpmml/jpmml-sparkml/releases/tag/1.1.23) |
| 2.1.X | [`1.2.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.2.X) | [`1.2.15`](https://github.com/jpmml/jpmml-sparkml/releases/tag/1.2.15) |
| 2.2.X | [`1.3.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.3.X) | [`1.3.15`](https://github.com/jpmml/jpmml-sparkml/releases/tag/1.3.15) |
| 2.3.X | [`1.4.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.4.X) | [`1.4.14`](https://github.com/jpmml/jpmml-sparkml/releases/tag/1.4.14) |
| 2.4.X | [`master`](https://github.com/jpmml/jpmml-sparkml/tree/master) | [`1.5.7`](https://github.com/jpmml/jpmml-sparkml/releases/tag/1.5.7) |

JPMML-SparkML checks the version of Apache Spark runtime environment before doing any conversion work.
For example, the following exception is thrown when JPMML-SparkML version 1.4(.14) discovers that it has been improperly paired with Apache Spark version 2.2:

```
java.lang.IllegalArgumentException: Expected Apache Spark ML version 2.3, got version 2.2 (2.2.0)
  at org.jpmml.sparkml.ConverterFactory.checkVersion(ConverterFactory.java:102)
  at org.jpmml.sparkml.PMMLBuilder.<init>(PMMLBuilder.java:81)
  ... 48 elided
```

### Library JAR

The library JAR file can be imported into Apache Spark version 2.3.0 (and newer) using the `--packages` command-line option. Package coordinates must follow Apache Maven conventions `${groupId}:${artifactId}:${version}`, where the groupId and artifactId are fixed as `org.jpmml` and `jpmml-sparkml`, respectively.

For example, starting Spark shell with the JPMML-SparkML library JAR:

```
$ export SPARK_HOME=/opt/spark-2.3.0/
$ $SPARK_HOME/bin/spark-shell --packages org.jpmml:jpmml-sparkml:${version}
```

**Important**: This library JAR file is not directly usable with Apache Spark versions 2.0 through 2.2 due to the [SPARK-15526](https://issues.apache.org/jira/browse/SPARK-15526) classpath conflict.

This classpath conflict typically manifests itself during the conversion work, in the form of some obscure `java.lang.NoSuchFieldError` or `java.lang.NoSuchMethodError`:

```
java.lang.NoSuchMethodError: org.dmg.pmml.Field.setOpType(Lorg/dmg/pmml/OpType;)Lorg/dmg/pmml/PMMLObject;
  at org.jpmml.converter.PMMLEncoder.toCategorical(PMMLEncoder.java:215)
  at org.jpmml.sparkml.feature.StringIndexerModelConverter.encodeFeatures(StringIndexerModelConverter.java:74)
  at org.jpmml.sparkml.FeatureConverter.registerFeatures(FeatureConverter.java:47)
  at org.jpmml.sparkml.feature.RFormulaModelConverter.registerFeatures(RFormulaModelConverter.java:65)
  at org.jpmml.sparkml.PMMLBuilder.build(PMMLBuilder.java:114)
  at org.jpmml.sparkml.PMMLBuilder.buildByteArray(PMMLBuilder.java:278)
  at org.jpmml.sparkml.PMMLBuilder.buildByteArray(PMMLBuilder.java:274)
  ... 48 elided
```

### Executable uber-JAR

The executable uber-JAR file can be imported into any Apache Spark version using the `--jars` command-line option.

For example, starting PySpark with the JPMML-SparkML executable uber-JAR:

```
$ export SPARK_HOME=/opt/spark-2.2.0/
$ $SPARK_HOME/bin/pyspark --jars /path/to/jpmml-sparkml-executable-${version}.jar
```

## Updating application code ##

The `org.jpmml.sparkml.ConverterUtil` utility class is still part of JPMML-SparkML, but it has been marked as deprecated, and rendered dysfunctional - both `#toPMML(StructType, PipelineModel)` and `#toPMMLByteArray(StructType, PipelineModel)` utility methods always throw an `java.lang.UnsupportedOperationException`:

```
java.lang.UnsupportedOperationException: Replace "org.jpmml.sparkml.ConverterUtil.toPMMLByteArray(schema, pipelineModel)" with "new org.jpmml.sparkml.PMMLBuilder(schema, pipelineModel).buildByteArray()"
  at org.jpmml.sparkml.ConverterUtil.toPMMLByteArray(ConverterUtil.java:51)
  at org.jpmml.sparkml.ConverterUtil.toPMMLByteArray(ConverterUtil.java:46)
  ... 48 elided
```

The exception message provides adequate instructions for updating the application code.

Old API:

``` java
StructType schema = df.schema();
PipelineModel pipelineModel = pipeline.fit(df);

byte[] pmmlBytes = ConverterUtil.toPMMLByteArray(schema, pipelineModel);
```

New API:

``` java
StructType schema = df.schema();
PipelineModel pipelineModel = pipeline.fit(df);

byte[] pmmlBytes = new PMMLBuilder(schema, pipelineModel).buildByteArray();
```

The `org.jpmml.sparkml.PMMLBuilder` class currently exposes three builder methods:

* `#build()` - Returns the PMML document as a live `org.dmg.pmml.PMML` object.
* `#buildByteArray()` - Returns the PMML document as a byte array.
* `#buildFile(java.io.File)` - Writes the PMML document to the specified file in local filesystem. Upon success, returns the argument `java.io.File` object unchanged.

The first option is aimed at PMML-savvy applications that wish to perform extra processing on the PMML document (eg. adding or removing feature transformations).
However, most applications should be content with the JPMML-SparkML generated PMML document, and will be processing it as a generic blob.
The choice between the last two options depends on the approximate size/complexity of the PMML document (eg. elementary models vs. ensemble models) and overall application architecture.

## New API features and functionality ##

### Conversion options

The "business logic" of some Apache Spark ML transformer or model can often be converted to the PMML representation in more than one way. Some representations are easier to approach for humans (eg. interpretation and manual modification), whereas some other representations are more compact or faster to execute for machines.

The purpose of conversion options is to activate the most optimal representation for the intended application scenario. Granted, the content of PMML documents is well structured and is fairly easy to manipulate using [JPMML-Model](https://github.com/jpmml/jpmml-model) and [JPMML-Converter](https://github.com/jpmml/jpmml-converter) libraries at any later time. However, achieving the desired outcome by toggling high-level controls is much more productive than writing low-level application code.

There is no easy recipe for deciding which conversion options to tweak, and in which way. It could very well be the case that the defaults work fine for everything except for one specific feature/operation/algorithmic detail.
The recommended way of going about this problematics is generating a grid of PMML documents by systematically varying the values of small number of key conversion options, and capturing and analyzing relevant metrics during evaluation.

Conversion options are systematized as Java marker interfaces, which inherit from the `org.jpmml.sparkml.HasOptions` base marker interface.
A similar convention is being enforced across all JPMML conversion libraries. For example, in the [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) library, which deals with the conversion of Scikit-Learn pipelines to the PMML representation, the class hierarchy is rooted at the `org.jpmml.sklearn.HasOptions` base marker interface.

Unfortunately, the documentation is severely lacking in this area.
To discover and learn which conversion options are available (in a particular JPMML-SparkML version), simply order the Java IDE to display the class hierarchy starting from the `HasOptions` base marker interface, and browse through it.

Notable conversion options:

* `org.jpmml.sparkml.model.HasRegressionOptions#OPTION_LOOKUP_THRESHOLD`. Controls the encoding of categorical features in regression models - table scan vs. table lookup.
* `org.jpmml.sparkml.model.HasTreeOptions#OPTION_COMPACT`. Controls the encoding of the tree data structure in decision tree models and their ensembles - Apache Spark-style binary splits vs. PMML-style multi-splits. Can reduce the size of PMML documents anywhere between 25 to 50%.

For maximum future-proofness, all conversion option names and values should be given as Java class constants. For example, the name of the decision tree compaction option should be given as `org.jpmml.sparkml.model.HasTreeOptions#OPTION_COMPACT` (instead of a Java string literal `"compact"`). If this conversion options should be renamed, relocated, or removed in some future JPMML-SparkML version, then the Java IDE/compiler would automatically issue a notification about it.

The `PMMLBuilder` class exposes the following mutator methods:

* `#putOption(String, Object)` - Sets the conversion option for all pipeline stages.
* `#putOption(PipelineStage, String, Object)` - Sets the conversion option for the specified pipeline stage only.

### Verification

Every conversion operation raises concern, whether the JPMML-SparkML library was doing a good job or not. Say, the conversion operation appears to have succeeded (ie. there were no exceptions thrown, and no warning- or error-level log messages emitted), but how to be sure that the PMML representation of the fitted pipeline shall be making exactly the same predictions as the Apache Spark representation did?

The PMML specification addresses this condundrum with the [model verification](https://dmg.org/pmml/v4-4-1/ModelVerification.html) mechanism.
In brief, it is possible to embed a verification data into models. The verification dataset has two column groups - the inputs, and the predictions that the original ML framework made when those inputs were fed into it.
PMML engines are expected to perform self-checks using the verification data before commissioning the model. If the live predictions made by the PMML engine agree with the stored predictions of the original ML framework (within the defined acceptability criteria), then that is a strong indication that everything is going favourably.
Next to most common cases, the verification dataset should aim to include all sorts of fringe cases (eg. missing, outlier, invalid values) in order to increase the confidence.

The [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library can be ordered to perform self-checks on the `org.jpmml.evaluator.Evaluator` object by invoking its `#verify()` method.
JPMML-SparkML integration tests indicate that the JPMML ecosystem (ie. JPMML-SparkML converter plus JPMML-Evaluator scorer) is consistently able to reproduce Apache Spark predictions (eg. regression targets, classification probabilities) with an abolute/relative error of 1e-14 or less.

The `PMMLBuilder` class exposes the following mutator methods:

* `#verify(Dataset<Row>)` - Embeds the verification dataset.
* `#verify(Dataset<Row>, double, double)` - Embeds the verification dataset with custom acceptance criteria (precision and zero threshold).

## JPMML-SparkML wrappers ##

The JPMML-SparkML library is written in Java. It is very easy to integrate into any Java or Scala application to give it Apache Spark ingestion capabilities.

However, there is an even more important audience of data scientists that would like to access this functionality from within their Python (PySpark) and R (SparkR and Sparklyr) scripts.

The JPMML ecosystem now includes Python and R wrapper libraries for the JPMML-SparkML library. The wrappers are kept as minimal and shallow as possible. Essentially, they provide a language-specific class that communicates with the underlying `org.jpmml.sparkml.PMMLBuilder` Java class, and handle the conversion of objects between the two environments (eg. converting Python and R strings to Java strings, and vice versa).

### PySpark

The [`pyspark2pmml`](https://github.com/jpmml/pyspark2pmml) package works with the official [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) interface.

The `pyspark2pmml.PMMLBuilder` Python class is effectively an API clone (in terms of the assortment and signatures of its methods) of the `org.jpmml.sparkml.PMMLBuilder` Java class.

The only noteworthy difference is that it has a three-argument constructor (instead of a two-argument one):

1. Apache Spark connection in the form of a [`pyspark.SparkContext`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.html) object.
2. Training dataset in the form of a [`pyspark.sql.DataFrame`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html) object.
3. Fitted pipeline in the form of a [`pyspark.ml.PipelineModel`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.PipelineModel.html) object.

The Apache Spark connection is typically available in PySpark session as the `sc` variable. The `SparkContext` class has an `_jvm` attribute, which gives Python (power-)users direct access to JPMML-SparkML functionality via the Py4J gateway.

Sample usage:

``` python
from pyspark.ml import Pipeline

df = ...
pipeline = Pipeline(stages = [...])

pipelineModel = pipeline.fit(df)

from pyspark2pmml import PMMLBuilder

pmmlBuilder = PMMLBuilder(sc, df, pipelineModel) \
  .putOption(None, sc._jvm.org.jpmml.sparkml.model.HasTreeOptions.OPTION_COMPACT, True) \
  .verify(df.sample(False, 0.01))

pmmlBuilder.buildFile("pipeline.pmml")
```

### Sparklyr

There is no package for the official [SparkR](https://spark.apache.org/docs/latest/sparkr.html) interface.
However, the [`sparklyr2pmml`](https://github.com/jpmml/sparklyr2pmml) package works with RStudio's [Sparklyr](https://spark.rstudio.com/) interface.

Contrary to Java, Scala and Python, object-oriented design and programming is a bit challenging in R.

The `sparklyr2pmml::PMMLBuilder` S4 class can only capture the state. It defines two slots `sc` and `java_pmml_builder`, which can be initialized via the two-argument constructor. However, most R users are advised to regard this constructor as private, and heed the three-argument `sparklyr2pmml::PMMLBuilder` S4 helper function instead. This function takes care of initializing a proper `java_pmml_builder` object based on the `sparklyr::tbl_spark` training dataset and `sparklyr::ml_pipeline_model` fitted pipeline objects.

All mutator and builder methods have been outsourced to standalone S4 generic functions, which take the `sparklyr2pmml::PMMLBuilder` object as the first argument:

* `putOption(PMMLBuilder, ml_pipeline_stage, character, object)`
* `verify(PMMLBuilder, tbl_spark)`
* `build(PMMLBuilder)`
* `buildByteArray(PMMLBuilder)`
* `buildFile(PMMLBuilder, character)`

Sample usage:

``` r
library("dplyr")
library("sparklyr")

df = ...
pipeline = ml_pipeline(sc) %>%
  ...

pipeline_model = ml_fit(pipeline, df)

library("sparklyr2pmml")

# Use `magrittr`-style %>% operators for chaining the constructor helper function and subsequent mutator functions together
pmml_builder = PMMLBuilder(sc, df, pipeline_model) %>%
  putOption(NULL, invoke_static(sc, "org.jpmml.sparkml.model.HasTreeOptions", "OPTION_COMPACT"), TRUE) %>%
  verify(sdf_sample(df, 0.01, replacement = FALSE))

buildFile(pmml_builder, "pipeline.pmml")
```
