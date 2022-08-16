---
layout: post
title: "Deploying R language models on Apache Spark ML"
author: vruusmann
keywords: r apache-spark r2pmml jpmml-evaluator
---

The [R platform](https://www.r-project.org/) provides a wider and deeper selection of algorithms than any other platform. The trouble is that all these algorithms are tightly coupled to the R language runtime and package system, which makes their reuse on other platforms and application environments fairly difficult.

This blog post demonstrates how to decouple a fitted R model from the R platform by converting it to the standardized Predictive Model Markup Language (PMML) representation. A PMML model is easy to achive and deploy across application environments. Here, the PMML model is turned into an Apache Spark ML transformer, which operates in Java/JVM memory space and can be easily dispatched to where the data resides.

### R side

The exercise starts with training a logistic regression model for the "audit" dataset.

``` r
audit = read.csv("audit.csv")
audit$Adjusted = as.factor(audit$Adjusted)

audit.formula = as.formula(
  Adjusted
  ~
  # Include all raw columns as a starting point
  .
  # Append interactions
  + Gender:Marital + Gender:Hours
  # Append the estimated hourly wage
  + I(Income / (Hours * 52))
  # Take out the raw "Age" column, and append a binned one
  - Age + base::cut(Age, breaks = c(0, 18, 65, 120))
  # Take out the raw "Employment" column, and append a re-mapped one
  - Employment + plyr::mapvalues(Employment, c("PSFederal", "PSState", "PSLocal"), c("Public", "Public", "Public"))
)

audit.glm = glm(audit.formula, data = audit, family = "binomial")
```

The R platform lacks the pipeline concept. Feature engineering can happen either as free-form R code (applying functions "step by step" to the dataset) or as model formula (combining functions to a master function and applying it "atomically" to the dataset).

The model formula approach requires a bit more experience and discipline to pull off. However, it has a clear and major advantage that the resulting R models are self-contained - all feature engineering logic is stored inside the model object, and is automatically executed whenever the model object is used with the standard `stats::predict(..)` function.

The [`r2pmml`](https://github.com/jpmml/r2pmml) package checks R models for model formula information, and if present, will analyze and convert it to the PMML representation as fully as possible.

Supported constructs:

* Interactions using the `:` operator.
* Free-form expressions and predicates using the `base::I(..)` function:
   * Logical operators `&`, `|` and `!`.
   * Relational operators `==`, `!=`, `<`, `<=`, `>=` and `>`.
   * Arithmetic operators `+`, `-`, `/` and `*`.
   * Exponentiation operators `^` and `**`.
   * Arithmetic functions `abs`, `ceiling`, `exp`, `floor`, `log`, `log10`, `round` and `sqrt`.
   * Missing value check function `is.na`.
* Conditional logic using the `base::ifelse(..)` function.
* Continuous feature binning using the `base::cut(..)` function.
* Categorical feature re-mapping using `plyr::revalue(..)` and `plyr::mapvalues(..)` functions.

There is always the doubt whether the `r2pmml` package did get everything right, meaning that the generated PMML model has the same input/output interface and is making the same predictions as the R model.

This doubt can be somewhat alleviated by manual inspection of the PMML document. For example, making sure that all "raw" input fields are correctly defined under the `/PMML/DataDictionary` element (name, type and the value domain), and all "derived" values under the `/PMML/TransformationDictionary` element.

It is possible to remove all doubts about the PMML model executability and correctness using the [model verification](http://dmg.org/pmml/v4-4-1/ModelVerification.html) mechanism:

``` r
library("dplyr")
library("r2pmml")

audit_sample = sample_n(audit, 10)
audit_sample$Adjusted = NULL

audit.glm = verify(audit.glm, newdata = audit_sample)
```

The idea behind model verification is to make predictions on a small but representative dataset (could be a subset of the training dataset, or some manually crafted dataset covering all known edge and corner cases) using the R model.

The `r2pmml::verify(obj, newdata)` function decorates the R model with a `verification` element. The R to PMML converter looks for this element, and if found, decorates the PMML model with a `ModelVerification` element.

The conversion functionality is available via the `r2pmml::r2pmml(obj, path)` package namesake function:

``` r
library("r2pmml")

r2pmml(audit.glm, "LogisticRegressionAudit.pmml")
```

### Apache Spark side

The [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library provides good all-purpose PMML engine for the Java/JVM platform. This library operates on individual data records, which must be translated back and forth to the `java.util.Map<FieldName, ?>` representation.

Apache Spark applications are much better off working with the [JPMML-Evaluator-Spark](https://github.com/jpmml/jpmml-evaluator-spark) library, which turns this low-level PMML engine into an already familiar high-level Apache Spark ML transformer (ie. [`org.apache.spark.ml.Transformer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/Transformer.html) subclass).

JPMML-Evaluator-Spark exists in two flavours:

* Library JAR file `jpmml-evaluator-spark-${version}.jar`. Contains `org.jpmml.evaluator.spark.*` classes. Distributed via the Maven Central repository.
* Runtime uber-JAR file `jpmml-evaluator-spark-runtime-${version}.jar`. Contains all library JAR file classes, plus all transitive dependency (JPMML-Evaluator, JPMML-Model, Google Guava, etc.) classes. Distributed via the [GitHub releases](https://github.com/jpmml/jpmml-evaluator-spark/releases) page.

The library JAR file can be imported into Apache Spark version 2.3.0 (and newer) using the `--packages` command-line option. Package coordinates must follow Apache Maven conventions `${groupId}:${artifactId}:${version}`, where the groupId and artifactId are fixed as `org.jpmml` and `jpmml-evaluator-spark`, respectively.

For example, starting Spark shell with the JPMML-Evaluator-Spark library JAR:

```
$ export SPARK_HOME=/opt/spark-2.3.0/
$ $SPARK_HOME/bin/spark-shell --packages org.jpmml:jpmml-evaluator-spark:${version}
```

**Important**: this library JAR file is not directly usable with Apache Spark versions 2.0 through 2.2 due to the [SPARK-15526](https://issues.apache.org/jira/browse/SPARK-15526) classpath conflict.

The PMML engine is created as usual. With the introduction of the builder pattern (available in JPMML-Evaluator version 1.4.5 and newer), it shouldn't take more than a couple lines of boilerplate code to build an `org.jpmml.evaluator.Evaluator` object based on a PMML byte stream or file.

``` scala
import java.io.File
import org.jpmml.evaluator.LoadingModelEvaluatorBuilder

val evaluatorBuilder = new LoadingModelEvaluatorBuilder() \
  .load(new File("LogisticRegressionAudit.pmml"))

val evaluator = evaluatorBuilder.build()

evaluator.verify()
```

The `Transformer` object can be created manually or using the `org.jpmml.evaluator.spark.TransformerBuilder` class. Model fields are typically mapped to Apache Spark dataset columns on a group basis using `TransformerBuilder#withTargetCols()` and `TransformerBuilder#withOutputCols()` configuration methods. However, if the model is known to follow a specific contract, then it is possible to map its fields individually using function-specific configuration methods.
For example, the probability distribution of a probabilistic classification model can be mapped to an Apache Spark ML-style vector column using the `TransformerBuilder#withProbabilityCol(String, List<String>)` configuration method.

``` scala
import org.jpmml.evaluator.spark.TransformerBuilder

val transformerBuilder = new TransformerBuilder(evaluator) \
  .withTargetCols() \
  .withOutputCols() \
  .exploded(true)

val transformer = transformerBuilder.build()
```

It should be pointed out that the JPMML-Evaluator-Spark library is developed in the Java language, and that its public API (eg. method signatures, return types) only makes use of Java types. This may necessitate extra type casts and/or conversions when working in other languages such as the Scala language.

The `Transformer` object holds the complete "business logic" of the above R script, including all feature engineering, model execution and decision engineering functionality. It also takes full care of translating values between Apache Spark and PMML type systems.

``` scala
var inputDs = spark.read.format("csv") \
  .option("header", "true") \
  .load("audit.csv")

// Drop the raw target column
inputDs = inputDs.drop("Adjusted")
inputDs.printSchema()

var resultDs = transformer.transform(inputDs)

// Select predicted target and output columns
resultDs = resultDs.select("Adjusted", "probability(0)", "probability(1)")
resultDs.printSchema()

resultDs.show(10)
```

For example, if the "audit" dataset is loaded from a CSV document without specifying `option("inferSchema", "true")`, then all columns default to the `java.lang.String` data type. A dummy or mismatching dataset schema is not a problem, because the underlying PMML engine automatically parses String values to correct PMML data type values, and proceeds with the rest of input value preparation workflow as usual.

Prediction columns are appended to the input dataset. Depending on the setting of the `TransformerBuilder#exploded(boolean)` configuration method, they are either appended collectively as a single `struct` column, or individually as many scalar columns.

### Resources

* "Audit" dataset: [`audit.csv`]({{ site.baseurl }}/assets/data/audit.csv)
* R script: [`train.R`]({{ site.baseurl }}/assets/2019-02-09/train.R)
* Scala script: [`deploy.scala`]({{ site.baseurl }}/assets/2019-02-09/deploy.scala)