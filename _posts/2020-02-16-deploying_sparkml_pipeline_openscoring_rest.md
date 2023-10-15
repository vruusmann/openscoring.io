---
layout: post
title: "Deploying Apache Spark ML pipelines on Openscoring REST web service"
author: vruusmann
keywords: apache-spark pyspark jpmml-sparkml pyspark2pmml openscoring
--- 

This blog post is a rehash of an earlier blog post about [using Apache Spark ML pipelines for real-time prediction]({% post_url 2016-07-04-sparkml_realtime_prediction_rest_approach %}).
It aims to demonstrate how things have evolved over the past 3.5 years, so that the proposed approach should now be intelligible to and executable by anyone with basic Apache Spark ML (PySpark flavour) experience.

The workflow has four steps:

1. Importing JPMML-SparkML library into Apache Spark.
2. Assembling and fitting a pipeline model, converting it to the PMML representation.
3. Starting Openscoring REST web service.
4. Using Python client library to work with Openscoring REST web service.

## Importing JPMML-SparkML into Apache Spark ##

The [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library converts Apache Spark ML pipeline models to the Predictive Model Markup Language (PMML) representation.

This library can be bundled statically with the application, or imported dynamically into the application driver program using `--jars` or `--packages` command-line options.

Users of Apache Spark 2.0, 2.1 and 2.2 are advised to download a suitable version of the JPMML-SparkML executable uber-JAR file from the [GitHub releases](https://github.com/jpmml/jpmml-sparkml/releases) page, and include it into their environment using the `--jars /path/to/jpmml-sparkml-executable-${version}.jar` command-line option.

For example, including JPMML-SparkML 1.3.15 into Apache Spark 2.2:

```
$ export SPARK_HOME=/opt/apache-spark-2.2.X
$ wget https://github.com/jpmml/jpmml-sparkml/releases/download/1.3.15/jpmml-sparkml-executable-1.3.15.jar
$ $SPARK_HOME/bin/pyspark --jars jpmml-sparkml-executable-1.3.15.jar
```

Users of Apache Spark 2.3, 2.4 and newer are advised to fetch the JPMML-SparkML library (plus its transitive dependencies) straight from the Maven Central repository using the `--packages org.jpmml:jpmml-sparkml:${version}` command-line option:

For example, including JPMML-SparkML 1.5.7 into Apache Spark 2.4:

```
$ export SPARK_HOME=/opt/apache-spark-2.4.X
$ $SPARK_HOME/bin/pyspark --packages org.jpmml:jpmml-sparkml:1.5.7
```

The JPMML-SparkML library is written in the Java language.

PySpark users should additionally install the [`pyspark2pmml`](https://github.com/jpmml/pyspark2pmml) package, which provides Python language wrappers for JPMML-SparkML public API classes and methods:

```
$ pip install --upgrade pyspark2pmml
```

## Assembling, fitting and converting pipeline models ##

The JPMML-SparkML library supports most common Apache Spark ML transformer and model types.

Selected highlights:

* Pipeline assembly via [`feature.RFormula`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RFormula.html).
* Feature engineering via [`feature.SQLTransformer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/SQLTransformer.html).
* Hyperparameter selection and tuning via [`tuning.CrossValidator`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/tuning/CrossValidator.html) and [`tuning.TrainValidationSplit`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/tuning/TrainValidationSplit.html).
* Third-party ML framework model types such as XGBoost and LightGBM (MMLSpark).
* Custom transformer and model types.

The exercise starts with training two separate classification-type decision tree models for the "red" and "white" subsets of the ["wine quality"](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) dataset.

For demonstration purposes, the original dataset is enriched with a "ratio of free sulfur dioxide" column by dividing the "free sulfur dioxide" column with the "total sulfur dioxide" column using Apache Spark SQL (by convention, column names must be surrounded with backticks if they contain whitespace):

``` python
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import RFormula, SQLTransformer

df = spark.read.option("delimiter", ";").csv("winequality-red.csv", header = True, inferSchema = True)

statement = """
  SELECT *,
  (`free sulfur dioxide` / `total sulfur dioxide`) AS `ratio of free sulfur dioxide`
  FROM __THIS__
"""
sqlTransformer = SQLTransformer(statement = statement)
formula = "quality ~ ."
rFormula = RFormula(formula = formula)
classifier = DecisionTreeClassifier(minInstancesPerNode = 20)
pipeline = Pipeline(stages = [sqlTransformer, rFormula, classifier])
pipelineModel = pipeline.fit(df)
```

The conversion of pipeline models is essentially a one-liner:

``` python
from pyspark2pmml import PMMLBuilder

PMMLBuilder(sc, df, pipelineModel) \
  .buildFile("RedWineQuality.pmml")
```

The `pyspark2pmml.PMMLBuilder` Python class is a thin wrapper around the `org.jpmml.sparkml.PMMLBuilder` Java class, and "inherits" the majority of its public API methods unchanged. 
It is possible to use `PMMLBuilder.putOption(stage: ml.PipelineStage, name, value)` and `PMMLBuilder.verify(df: sql.DataSet)` methods to configure the look and feel of PMML markup and embed model verification data, respectively, as described in an earlier blog post about [converting Apache Spark ML pipelines to PMML]({% post_url 2018-07-09-converting_sparkml_pipeline_pmml %}).

For demonstration purposes, disabling decision tree compaction (replaces binary splits with multi-way splits), and embedding five randomly chosen data records as model verification data:

``` python
from pyspark2pmml import PMMLBuilder

PMMLBuilder(sc, df, pipelineModel) \
  .putOption(classifier, "compact", False) \
  .putOption(classifier, "keep_predictionCol", False) \
  .verify(df.sample(False, 0.005).limit(5)) \
  .buildFile("RedWineQuality.pmml"))
```

Unlike any other ML persistence or serialization data format, the PMML data format is text based and designed to be human-readable.
It is possible to open the resulting `RedWineQuality.pmml` and `WhiteWineQuality.pmml` files in a text editor and follow the splitting logic of the learned decision tree models in terms of the original feature space.

## Starting Openscoring REST web service ##

The quickest way to have something happening is to download the latest Openscoring server executable uber-JAR file from the [GitHub releases](https://github.com/openscoring/openscoring/releases) page, and run it.

For example, running Openscoring standalone server 2.0.1:

```
$ wget https://github.com/openscoring/openscoring/releases/download/2.0.1/openscoring-server-executable-2.0.1.jar
$ java -jar openscoring-server-executable-2.0.1.jar
```

There should be a Model REST API endpoint ready at [http://localhost:8080/openscoring/model](http://localhost:8080/openscoring/model) now.
The default user authorization logic is implemented by the `org.openscoring.service.filters.NetworkSecurityContextFilter` JAX-RS filter class, which grants "user" role (read-only) to any address and "admin" role (read and write) to local host addresses.

When looking to upgrade to a more production-like setup, then [Openscoring-Docker](https://github.com/openscoring/openscoring-docker) and [Openscoring-Elastic-Beanstalk](https://github.com/openscoring/openscoring-elastic-beanstalk) projects provide good starting points.

## Using Python client library to work with Openscoring REST web service ##

The [Openscoring REST API](https://github.com/openscoring/openscoring#rest-api) is simple and straightforward.

Nevertheless, Python users should install the [`openscoring`](https://github.com/openscoring/openscoring-python) package that provides an even simpler high-level API.

```
$ pip install --upgrade openscoring
```

The `openscoring.Openscoring` class holds common information such as the REST API base URL, credentials etc.

The base URL is this part of URL that is shared between all endpoints.
It typically follows the pattern `http://<server>:<port>/<context path>`.
The Openscoring standalone server uses a non-empty context path `openscoring` for disambiguation purposes, so the default base URL is `http://localhost:8080/openscoring`.

``` python
from openscoring import Openscoring

os = Openscoring("https://localhost:8080/openscoring")
```

A single Openscoring application instance can host multiple models.
Individual models are directly addressable in the REST API by appending a slash and their alphanumeric identifier to the URL of the Model REST API endpoint.

``` python
# Shall be available at http://localhost:8080/openscoring/model/RedWineQuality
os.deployFile("RedWineQuality", "RedWineQuality.pmml")

# Shall be available at http://localhost:8080/openscoring/model/WhiteWineQuality
os.deployFile("WhiteWineQuality", "WhiteWineQuality.pmml")
```

It is advisable to open model URLs in a browser and examine the model schema description part (names, data types and value spaces of all input, target and output fields) of the response object.

For example, the model schema for "RedWineQuality" lists seven input fields, one target field and eight output fields.
It follows that this model does not care about four input fields (ie. "fixed acidity", "citric acid", "chlorides" and "density" columns) that were present in the `winequality-red.csv` dataset.
The mappings for these input fields may be safely omitted when making evaluation requests:

``` python
dictRequest = {
  #"fixed acidity" : 7.4,
  "volatile acidity" : 0.7,
  #"citric acid" : 0,
  "residual sugar" : 1.9,
  #"chlorides" : 0.076,
  "free sulfur dioxide" : 11,
  "total sulfur dioxide" : 34,
  #"density" : 0.9978,
  "pH" : 3.51,
  "sulphates" : 0.56,
  "alcohol" : 9.4,
}

dictResponse = os.evaluate("RedWineQuality", dictRequest)
print(dictResponse)
```

The "single prediction" mode is intended for real-time application scenarios.
Openscoring uses the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library as its PMML engine, and should be able to deliver sub-millisecond turnaround times for arbitrary complexity PMML documents.

The "batch prediction" mode is intended for application scenarios, where new data becomes available at regular intervals, or where the cost of transporting data over the computer network (eg. calling a service from remote locations) is the limiting factor:

``` python
import pandas

dfRequest = pandas.read_csv("winequality-white.csv", sep = ";")

dfResponse = os.evaluateCsv("WhiteWineQuality", dfRequest)
print(dfResponse.head(5))
```

When a model is no longer needed, then it should be undeployed to free up server resources:

``` python
os.undeploy("RedWineQuality")
os.undeploy("WhiteWineQuality")
```

## Resources ##

* "Wine quality" dataset: [`winequality-red.csv`](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) and [`winequality-white.csv`](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv)
* Python scripts: [`train.py`]({{ "/resources/2020-02-16/train.py" | absolute_url }}) and [`deploy.py`]({{ "/resources/2020-02-16/deploy.py" | absolute_url }})
