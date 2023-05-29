---
layout: post
title: "Training PySpark LightGBM pipelines"
author: vruusmann
keywords: apache-spark pyspark synapseml data-categorical data-invalid data-missing
---

LightGBM supports "distributed learning" mode, where the training of a single model is split between multiple computers.
Apache Spark users have the best-in-kind access to it using the [SynapseML](https://github.com/microsoft/SynapseML) (formerly MMLSpark) middleware library.

However, pushing LightGBM to its fullest potential in custom environments remains challenging.
This blog post demonstrates how to build PySpark pipelines for complex real-life datasets so that their key aspects (categorical features, missing values) are correctly presented.

## Setup ##

The PySpark LightGBM software stack has three major components:
1. LightGBM C++ library.
2. SynapseML Java wrapper library, which provides Apache Spark API for commanding #1.
3. SynapseML Python wrapper library, which provides PySpark API for commanding #2.

The installation is split into two units.
Namely, components #1 and #2 are packaged and distributed together as a Apache Spark package, whereas component #3 is a standalone Python package.

### Apache Spark side

Installing and activating the `com.microsoft.azure:synapseml-lightgbm_2.12` package:

```
$ $SPARK_HOME/bin/spark-submit --packages "com.microsoft.azure:synapseml-lightgbm_2.12:${synapseml.version}" main.py
```

At the time of writing this (May 2023), the Maven Central repository contains [six versions of this artifact](https://search.maven.org/artifact/com.microsoft.azure/synapseml-lightgbm_2.12):

| SynapseML version | LightGBM version |
|-------------------|------------------|
| 0.9.5 | 3.2.1 |
| 0.10.0 | 3.2.1 |
| 0.10.1 | 3.2.1 |
| 0.10.2 | 3.2.1 |
| 0.11.0 | 3.3.3 |
| 0.11.1 | 3.3.5 |

It can be seen that SynapseML generations 0.9 and 0.10 depend on LightGBM version 3.2.1, whereas SynapseML generation 0.11 depends on newer LightGBM versions 3.3.3 and 3.3.5.
However, from the data science perspective, they all should be more or less functionally equivalent.

### Python side

Installing the `synapseml` package:

```
$ python -m pip install synapseml==${synapseml.version}
```

The safest option would be to use identical SynapseML package versions on both sides.

Checking the installation:

``` python
import pyspark

print("PySpark version: {}".format(pyspark.__version__))

from pyspark.sql import SparkSession

spark = SparkSession.builder \
  .getOrCreate()

sc = spark.sparkContext
print("Spark version: {}".format(sc.version))

import synapse.ml.lightgbm as sml_lightgbm

print("SynapseML version: {}".format(sml_lightgbm.__version__))
```

The SynapseML Python wrapper library can be proven by loading the `synapse.ml.lightgbm` module, and querying its `__version__` attribute.
However, this does not convey any information about the underlying library layers.

Next, the SynapseML Java wrapper library can be proven by attempting to use some Java-backed functionality, such as constructing a dummy estimator:

``` python
from synapse.ml.lightgbm import LightGBMClassifier

classifier = LightGBMClassifier()
print(classifier)
```

If the Apache Spark package is not active, then this will fail with a characteristic Py4J type error `TypeError: 'JavaPackage' object is not callable`.

The version of the SynapseML Java wrapper library in use is not directly queriable.
With newer PySpark versions, it can be deduced by [listing Apache Spark resource files](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.listFiles.html#pyspark.SparkContext.listFiles):

``` python
sc = spark.sparkContext

# Requires PySpark 3.4 or newer
if hasattr(sc, "listFiles"):
  synapsemlResourceFiles = [scFile for scFile in sc.listFiles if "synapseml" in scFile]
  print("Spark SynapseML resource files: {}".format(synapsemlResourceFiles))
```

Alternatively, it can be noted from the Apache Spark log as the `SynapseMLLogInfo.buildVersion` entry:

```
23/05/26 10:25:23 INFO LightGBMClassifier: metrics/ {"buildVersion":"0.10.2","className":"class com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier","method":"constructor","uid":"LightGBMClassifier_903c5f1b3b2e"}
```

Finally, the LightGBM C++ library could be proven by fitting the dummy estimator.
However, given the tight physical and logical coupling between components #1 and #2, this check is likely to succeed at all times.

## Training ##

Pipeline template:

``` python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier

cat_cols = [...]
cont_cols = [...]

labelIndexer = StringIndexer(...)
labelIndexerModel = labelIndexer.fit(df)

catColumnsIndexer = StringIndexer(inputCols = cat_cols, outputCols = ["idx" + cat_col for cat_col in cat_cols])

vectorAssembler = VectorAssembler(inputCols = catColumnsIndexer.getOutputCols() + cont_cols, outputCol = "featureVector")

classifier = LightGBMClassifier(objective = "binary", numIterations = 117, labelCol = labelIndexerModel.getOutputCol(), featuresCol = vectorAssembler.getOutputCol())

pipeline = Pipeline(stages = [labelIndexerModel, catColumnsIndexer, vectorAssembler, classifier])
pipelineModel = pipeline.fit(df)
```

### Categorical data

LightGBM can do both continuous and categorical splits.

Categorical splits are attempted on integer columns that have been tagged as categorical.
The simplest way to perform such data pre-processing is using the [`StringIndexer`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html) transformer.

The tagging is based on column metadata.
Specifically, `StringIndexer` output columns carry a `ml_attr` tag, which signals that these double values represent array indices rather than generic data.
The SynapseML Java wrapper library must be relying on this information when auto-detecting the list of categorical features.

The auto-detection algorithm is disabled when this list was already declared during LightGBM model construction.

There are two declaration approaches possible.
First, in the name-based approach, the data scientist assigns names to all feature vector elements using the `slotNames` parameter, and then identifies the categorical subset using the `categoricalSlotNames` parameter:

``` python
classifier = LightGBMClassifier(slotNames = cat_cols + cont_cols, categoricalSlotNames = cat_cols, ...)
```

Second, in the index-based approach, the data scientist identifies the indices of categorical feature vector elements directly using the `categoricalSlotIndexes` parameter:

``` python
cat_col_indices = list(range(0, len(cat_cols)))

classifier = LightGBMClassifier(categoricalSlotIndexes = cat_col_indices, ...)
```

After fitting a LightGBM model, it is advisable to inspect its internal structure to see if all categorical features were detected and handled as such.
Getting the operational type of a feature wrong may do serious damage to its interpretability and predictive performance.

People who are more familiar with LightGBM internals can export the booster object into a text file, and inspect its header section for feature definitions as expressed in terms of `feature_names` and `feature_infos` entries.
Categorical feature infos match the `-1:<value_1>:<value_2>:..:<value_n>` pattern (ie. colon-separated enumeration of possible integer values), whereas continuous feature infos match the `[<value_min>:<value_max>]` pattern (ie. colon-separated range bounds).

The SynapseML exported booster text file does not include any helper sections that would elaborate the mapping of category levels between `StringIndexer` input and output columns.
This deficiency makes the migration of booster objects between application environments difficult.

If the booster object needs to be shared with Scikit-Learn, then the workaround will be to generate and append a `pandas_categorical`-style helper section to the booster text file.

### Sparse data

LightGBM generates binary splits, and tags one of the two branches as the "default branch".
If a data record contains fields with missing values, then they are not evaluated against the actual split condition, but are directly assigned to the default branch.
The same treatment applies to invalid values.

Technically speaking, there are three kinds of value spaces possible:
* Valid. A non-missing value, which was seen in the training dataset.
* Invalid. A non-missing value in a validation or testing dataset, which was not seen in the training dataset.
* Missing.

To rehash, a model is fitted on a dataset that contains valid and, possibly, missing values.
However, it can be used for prediction on datasets that contain all valid, missing and invalid values.

Popular ML frameworks such as Scikit-Learn and Apache Spark enforce a naivist approach, where data pre-processing transformers must perform data validation so that incoming non-valid values (ie. missing and invalid values) are transformed to valid values.

For example, in PySpark pipelines, missing values are stopped already at the forefront by replacing them with constant values (eg. sample mean, median or mode) using the [`Imputer`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html) transformer.
All the subsequent transformers act under the assumption that they will never encounter a missing value.

Adding support for missing and invalid values is all about getting Apache Spark's built-in data validation logic out of the way.
All values should be allowed to pass from one stage to another, up until the final model stage.

The fix is centered around setting the `handleInvalid` attribute of transformers from "error" to "keep":

``` python
vectorAssembler = VectorAssembler(inputCols = catColumnsIndexer.getOutputCols() + cont_cols, outputCol = "featureVector", handleInvalid = "keep")
```

The "keep" treatment is very effective with transformers that deal with numeric values.
However, the same cannot be said about various Apache Spark encoders and discretizers that deal with non-numeric values, because their transformation behaviour is more intrusive than it needs to be.

The case in point is once again the `StringIndexer` transformer.
It is expected that the "keep" treatment should transform `None` values to `NaN` values (string columns), or let `NaN` values pass through as-is (double columns).
In reality, the `StringIndexer` transformer maps all non-valid values to a special additional bucket, at index `len(labels)`.

The desired transformation behaviour would be to emit a `NaN` value or some negative integer value (ie. something that is easily distinguishable from "legible" array indices).
Unfortunately, the `StringIndexer` transformer cannot be configured to behave this way.

The workaround is to move the encoding of categorical features completely out of the main pipeline:

``` python
catColumnsIndexer = StringIndexer(inputCols = cat_cols, outputCols = ["idx" + cat_col for cat_col in cat_cols], handleInvalid = "keep")
catColumnsIndexerModel = catColumnsIndexer.fit(df)

df = catColumnsIndexerModel.transform(df)

# Replace the maximum value for each output column with -999 
for outputCol, labels in zip(catColumnsIndexerModel.getOutputCols(), catColumnsIndexerModel.labelsArray):
  df = df.replace(to_replace = float(len(labels)), value = -999, subset = [outputCol])
```

LightGBM will canonicalize all replacement values to `NaN` values, as made evident by Apache Spark log messages `[LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN`.
There is exactly one such log message being printed per column.

The `DataFrame.replace()` method appears to re-create columns (rather than updating values in place), because all `ml_attr` tags are lost.
This means that the list of categorical features cannot be auto-detected under no circumstances.
It must be declared during LightGBM model construction, using either the name-based or index-based approach (see the "categorical data" section above).

The training succeeds with all Apache Spark and SynapseML version combinations.

## Persistence ##

Fitted `PipelineModel` objects can be persisted for later deployment(s) using the `save()` method.

Unfortunately, Apache Spark versions 3.0.X and 3.1.X do not support the saving of embedded LightGBM models due to the following Py4J Java error:

```
Traceback (most recent call last):
  File "train.py", line 28, in <module>
    pipelineModel.save("LightGBMAudit")
  File "/opt/spark-3.0.3/python/lib/pyspark.zip/pyspark/ml/util.py", line 175, in save
  File "/opt/spark-3.0.3/python/lib/py4j-0.10.9-src.zip/py4j/java_gateway.py", line 1304, in __call__
  File "/opt/spark-3.0.3/python/lib/pyspark.zip/pyspark/sql/utils.py", line 128, in deco
  File "/opt/spark-3.0.3/python/lib/py4j-0.10.9-src.zip/py4j/protocol.py", line 326, in get_return_value
py4j.protocol.Py4JJavaError: An error occurred while calling o533.save.
: java.lang.NoClassDefFoundError: org/json4s/JsonListAssoc$
    at org.apache.spark.ml.ComplexParamsWriter$.getMetadataToSave(ComplexParamsSerializer.scala:126)
    at org.apache.spark.ml.ComplexParamsWriter$.saveMetadata(ComplexParamsSerializer.scala:97)
    at org.apache.spark.ml.ComplexParamsWriter.saveImpl(ComplexParamsSerializer.scala:40)
    at org.apache.spark.ml.util.MLWriter.save(ReadWrite.scala:168)
    ...
Caused by: java.lang.ClassNotFoundException: org.json4s.JsonListAssoc$
    at java.net.URLClassLoader.findClass(URLClassLoader.java:382)
    at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
    at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
    ... 43 more
```

The SynapseML Java wrapper library has been compiled against a [JSON4S](https://github.com/json4s/json4s) library version that is newer than the one(s) that is bundled with the Apache Spark installation.

This classpath issue cannot be resolved by end users.

The workaround is to extract the LightGBM model from the pipeline, and save it separately as a booster text file:

``` python
from pyspark.ml import PipelineModel

# The Apache Spark part - all stages except for the final model stage
preprocPipelineModel = PipelineModel(pipelineModel.stages[:-1])
preprocPipelineModel.save("PipelineModel")

# The SynapseML part - the final model stage
lgbmModel = pipelineModel.stages[-1]
lgbmModel.saveNativeModel("LightGBMClassificationModel")
```

Loading:

``` python
from synapse.ml.lightgbm import LightGBMClassificationModel

preprocPipelineModel = PipelineModel.load("PipelineModel")

lgbmModel = LightGBMClassificationModel.loadNativeModelFromFile("LightGBMClassificationModel")

# Restore optional attributes
lgbmModel.setLabelCol(labelIndexerModel.getOutputCol())
lgbmModel.setFeaturesCol(vectorAssembler.getOutputCol())

pipelineModel = PipelineModel(stages = preprocPipelineModel.stages + [lgbmModel])
```

The original pipeline and its saved-and-loaded clone make identical predictions.

## Resources ##

* Datasets: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }}) and [`audit-NA.csv`]({{ "/resources/data/audit-NA.csv" | absolute_url }})
* Python scripts: [`check.py`]({{ "/resources/2023-05-26/check.py" | absolute_url }}), [`train.py`]({{ "/resources/2023-05-26/train.py" | absolute_url }}) and [`train-NA.py`]({{ "/resources/2023-05-26/train-NA.py" | absolute_url }})
