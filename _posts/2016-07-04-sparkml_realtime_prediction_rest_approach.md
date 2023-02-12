---
layout: post
title: "Using Apache Spark ML pipeline models for real-time prediction: the Openscoring REST web service approach"
author: vruusmann
keywords: apache-spark openscoring
---

[Apache Spark](https://spark.apache.org/) follows the batch data processing paradigm, which has its strengths and weaknesses.
On one hand, the batch processing is suitable for working with Big Data-scale datasets. Apache Spark splits the task into manageable-size batches and distributes the workfload across a cluster of machines.
Apache Spark competitors such as R or Python cannot match that, because they typically require the task to fit into the RAM of a single machine.

On the other hand, the batch processing is characterized by high inertia. Apache Spark falls short in application scenarios where it is necessary to work with small datasets (eg. single data records) in real time.
Essentially, there is a lower bound (instead of an upper bound) to the effective size of a task.

This blog post details a workflow where Apache Spark ML pipeline models are converted to the Predictive Model Markup Language (PMML) representation, and then deployed using the Openscoring REST web service for easy interfacing with third-party applications.

### Converting Apache Spark ML pipeline models to the PMML representation

[PMML support](https://www.databricks.com/blog/2015/07/02/pmml-support-in-apache-spark-mllib.html) was introduced in Apache Spark MLlib version 1.4.0 in the form of a `org.apache.spark.mllib.pmml.PMMLExportable` trait. The invocation of the `PMMLExportable#toPMML()` method (or one of its overloaded variants) produces a PMML document withich contains the symbolic description of the fitted model.

Unfortunately, this solution is not very relevant anymore.
First, Apache Spark ML is organized around the pipeline formalization. A pipeline can be regarded as a directed graph of data transformations and models. When converting a model, then it will be necessary to include all the preceding pipeline stages to the dump.
Second, Apache Spark ML comes with rich metadata. The `DataFrame` representation of a dataset is associated with a static schema, which can be queried for column names, data types and more.
Finally, Apache Spark ML has replaced and/or abstracted away a great deal of Apache Spark MLlib APIs. Newer versions of Apache Spark ML have almost completely ceased to rely on Apache Spark MLlib classes that implement the `PMMLExportable` trait.

The [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library is an independent effort to provide a fully-featured PMML converter for Apache Spark ML pipelines.

The main interaction point is the `org.jpmml.sparkml.ConverterUtil#toPMML(StructType, PipelineModel)` utility method.
The conversion engine initializes a PMML document based on the `StructType` argument, and fills it with relevant content by iterating over all the pipeline stages of the `PipelineModel` argument.

The conversion engine requires a valid class mapping from `org.apache.spark.ml.Transformer` to `org.jpmml.sparkml.TransformerConverter` for every pipeline stage class.
The class mappings registry is automatically populated for most common transformer and model types. Application developers can implement and register their own `TransformerConverter` classes when looking to move beyond that.

Typical usage:

``` java
DataFrame dataFrame = ...;
StructType schema = dataFrame.schema();

Pipeline pipeline = ...;
PipelineModel pipelineModel = pipeline.fit(dataFrame);

PMML pmml = ConverterUtil.toPMML(schema, pipelineModel);

JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
```

The JPMML-SparkML library depends on a newer version of the [JPMML-Model](https://github.com/jpmml/jpmml-model) library than Apache Spark, which introduces severe compile-time and run-time classpath conflicts. The solution is to employ [Maven Shade Plugin](https://maven.apache.org/plugins/maven-shade-plugin/) and relocate the affected `org.dmg.pmml` and `org.jpmml.(agent|model|schema)` packages.

The [JPMML-SparkML-Bootstrap](https://github.com/jpmml/jpmml-sparkml-bootstrap) project aims to provide a complete example about developing and packaging an JPMML-SparkML powered application.

The `org.jpmml.sparkml.bootstrap.Main` application demonstrates a two-stage pipeline. The first pipeline stage is a `RFormula` transformer that simply selects columns from a CSV input file. The second pipeline stage is either a `DecisionTreeRegressor` or `DecisionTreeClassifier` predictor that finds the best approximation between the label column and feature columns. The result is written to a PMML output file.

The exercise starts with training a classification-type decision tree model for the ["wine quality" dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality):

```
$ $SPARK_HOME/bin/spark-submit \
  --class org.jpmml.sparkml.bootstrap.Main \
  /path/to/jpmml-sparkml-bootstrap/target/bootstrap-1.0-SNAPSHOT.jar \
  --formula "color ~ . -quality" \
  --csv-input /path/to/jpmml-sparkml-bootstrap/src/test/resources/wine.csv \
  --function CLASSIFICATION \
  --pmml-output wine-color.pmml
```

The resulting `wine-color.pmml` file can be opened for inspection in a text editor.

### The essentials of the PMML representation

A PMML document specifies a workflow for transforming an input data record to an output data record.
The end user interacts with the entry and exit interfaces of the workflow, and can completely disregard its internals.

The design and implementation of these two interfaces is PMML engine-specific. The [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library is geared towards maximum automation. The entry interface exposes complete description of active fields. Similarly, the exit interface exposes complete description of the primary target field and secondary output fields. A capable agent can use this information to format input data record and parse output data records without any external help.

##### Input

The decision tree model is represented as the `/PMML/TreeModel` element. Its schema is defined by the combination of `MiningSchema` and `Output` child elements.

A `MiningField` element serves as a collection of "import" and "export" statements. It refers to some field, and stipulates its role and requirements in the context of the current model element. The fields themselves are declared as `/PMML/DataDictionary/DataField` and `/PMML/TransformationDictionary/DerivedField` elements.

The wine color model defines eight input fields ("fixed_acidity", "volatile_acidity", .., "sulphates"). The values of input fields are prepared by performing type conversion from the user-specified representation to the PMML representation, which is followed by categorization into valid, invalid or missing subspaces, and application of subspace-specific treatments.

The default definition of the "fixed_acidity" input field:

``` xml
<PMML>
  <DataDictionary>
    <DataField name="fixed_acidity" optype="continuous" dataType="double"/>
  </DataDictionary>
  <TreeModel>
    <MiningSchema>
      <MiningField name="fixed_acidity"/>
    </MiningSchema>
  </TreeModel>
</PMML>
```

The same, after manual enhancement:

``` xml
<PMML>
  <DataDictionary>
    <DataField name="fixed_acidity" optype="continuous" dataType="double">
      <Value value="?" property="missing"/>
      <Interval closure="closure" leftMargin="3.8" rightMargin="15.9"/>
    </DataField>
  </DataDictionary>
  <TreeModel>
    <MiningSchema>
      <MiningField name="fixed_acidity" invalidValueTreatment="returnInvalid" missingValueReplacement="7.215307" missingValueTreatment="asMean"/>
    </MiningSchema>
  </TreeModel>
</PMML>
```

The enhanced definition reads:

1. If the user didn't supply a value for the "fixed_acidity" input field, or its string representation is equal to string constant "?", then replace it with string constant "7.215307".
2. Convert the value to `double` data type and `continuous` operational type.
3. If the value is in range [3.8, 15.9], then pass it on to the model element. Otherwise, throw an "invalid value" exception.

##### Output

The primary target field may be accompanied by a set of secondary output fields, which expose additional details about the prediction.
For example, classification models typically return the label of the winning class as the primary result, and the breakdown of the class probability distribution as the secondary result.

Secondary output fields are declared as `Output/OutputField` elements.

Apache Spark ML models indicate the availability of additional details by implementing marker interfaces.
The conversion engine keeps an eye out for the `org.apache.spark.ml.param.shared.HasProbabilityCol` marker interface. It is considered a proof that the classification model is capable of estimating class probability distribution, which is a prerequisite for encoding an `Output` element that contains probability-type `OutputField` child elements.

The wine color model defines a primary target field ("color"), and two secondary output fields ("probability_white" and "probability_red"):

``` xml
<PMML>
  <DataDictionary>
    <DataField name="color" optype="categorical" dataType="string">
      <Value value="white"/>
      <Value value="red"/>
    </DataField>
  </DataDictionary>
  <TreeModel>
    <MiningSchema>
      <MiningField name="color" usageType="target"/>
    </MiningSchema>
    <Output>
      <OutputField name="probability_white" feature="probability" value="white"/>
      <OutputField name="probability_red" feature="probability" value="red"/>
    </Output>
  </TreeModel>
</PMML>
```

In case of decision tree models, it is often desirable to obtain information about the decision path.
The identifier of the winning decision tree leaf can be queried by declaring an extra entityId-type `OutputField` element:

``` xml
<PMML>
  <TreeModel>
    <Output>
      <OutputField name="winnerId" feature="entityId"/>
    </Output>
  </TreeModel>
</PMML>
```

Apache Spark does not assign explicit identifiers to decision tree nodes.
Therefore, a PMML engine would be returning implicit identifiers in the form of a 1-based index, which are perfectly adequate for distinguishing between winning decision tree leafs.

The JPMML-Evaluator and JPMML-Model libraries provides rich APIs that can resolve node identifiers to `org.dmg.pmml.Node` class model objects, and backtrack these to the root of the decision tree.

##### Transformations

From the PMML perspective, Apache Spark ML data transformations can be classified as "real" or "pseudo".
A "real" transformation performs a computation on a feature or a feature vector. It is encoded as one or more `/PMML/DataDictionary/DerivedField` elements.

Examples of "real" transformers:

* `Binarizer`
* `Bucketizer`
* `MinMaxScaler`
* `PCA`
* `QuantileDiscretizer`
* `StandardScaler`

A `Binarizer` transformer for "discretizing" wine samples based on their sweetness:

``` java
Binarizer sweetnessBinarizer = new Binarizer()
  .setThreshold(6)
  .setInputCol("residual_sugar")
  .setOutputColumn("sweet_indicator");
```

The above, after conversion to the PMML representation:

``` xml
<PMML>
  <DataDictionary>
    <DerivedField name="sweet_indicator" dataType="double" optype="continuous">
      <Apply function="if">
        <Apply function="lessOrEqual">
          <FieldRef field="residual_sugar"/>
          <Constant>6.0</Constant>
        </Apply>
        <Constant>0.0</Constant>
        <Constant>1.0</Constant>
      </Apply>
    </DerivedField>
  </DataDictionary>
</PMML>
```

A "pseudo" transformation performs Apache Spark ML-specific housekeeping work such as assembling, disassembling or subsetting feature vectors.

Examples of "pseudo" transformers:

* `ChiSqSelector`
* `IndexToString`
* `OneHotEncoder`
* `RFormula`
* `StringIndexer`
* `VectorAssembler`
* `VectorSlicer`

The conversion engine is capable of performing smart analyses and optimizations in order to produce a maximally compact and expressive PMML document.
The case in point is the identification and pruning of unused field declarations, which improves the robustness and performance of production workflows

For example, the `wine.csv` CSV document contains 11 feature columns, but the wine color model reveals that three of them ("residual_sugar", "free_sulfur_dioxide" and "alcohol") do not contribute to the discrimination between white and red wines in any way.
The conversion engine takes notice of that and omits all the related data transformations from the workflow, thereby eliminating three-elevenths of the complexity.

### Importing PMML to Openscoring REST web service

[Openscoring](https://github.com/openscoring/openscoring) provides a way to expose an ML model as a REST web service.
The primary design consideration is to make ML models easily discoverable and usable (a variation of the [HATEOAS](https://en.wikipedia.org/wiki/HATEOAS) theme) for human and machine agents alike. 
The PMML representation is perfect fit thanks to the availability of rich descriptive metadata. Other representations can be plugged into the framework with the help of wrappers that satisfy the requested metadata query needs.

Openscoring is minimalistic Java web application that conforms to Servlet and JAX-RS specifications.

It can be built from the source checkout using [Apache Maven](https://maven.apache.org/):

```
$ git clone https://github.com/openscoring/openscoring.git
$ cd openscoring
$ mvn clean package
```

Openscoring exists in two variants. First, the standalone command-line application variant `openscoring-server/target/server-executable-${version}.jar` is based on Jetty web server. Easy configuration and almost instant startup and shutdown times make it suitable for local development and testing use cases. The web application (WAR) variant `openscoring-webapp/target/openscoring-webapp-${version}.war` is more suitable for production use cases. It can be deployed on any standards-compliant Java web- or application container, and secured and scaled according to organization's preferences.

Alternatively, release versions of the Openscoring WAR file can be downloaded from the [`org/openscoring/openscoring-webapp`](https://central.maven.org/maven2/org/openscoring/openscoring-webapp/) section of the Maven Central repository.

A demo instance of Openscoring can be launched by dropping its WAR file into the auto-deployment directory of a running [Apache Tomcat](https://tomcat.apache.org/) web container:

1. Download the latest `openscoring-webapp-${version}.war` file from the Maven Central repository to a temporary directory. At the time of writing this, it is [`openscoring-webapp-1.2.15.war`](https://central.maven.org/maven2/org/openscoring/openscoring-webapp/1.2.15/openscoring-webapp-1.2.15.war).
2. Rename the downloaded file to `openscoring.war`. Apache Tomcat generates the context path for a web application from the filename part of the WAR file. So, the context path for `openscoring.war` will be "/openscoring/" (whereas for the original `openscoring-webapp-${version}.war` it would have been "/openscoring-webapp-${version}/").
3. Move the `openscoring.war` file from the temporary directory to the `$CATALINA_HOME/webapps` auto-deployment directory. Allow the directory watchdog thread a couple of seconds to unpack and deploy the web application.
4. Verify the deployment by accessing [http://localhost:8080/openscoring/model](http://localhost:8080/openscoring/model). Upon success, the response body should be an empty JSON object `{ }`.

Openscoring maps every PMML document to a `/model/${id}` endpoint, which provides model-oriented information and services according to the [REST API specification](https://github.com/openscoring/openscoring#rest-api).

Model deployment, download and undeployment are privileged actions that are only accessible to users with the "admin" role. All the unprivileged actions are accessible to all users.
This basic access and authorization control can be overriden at the Java web container level. For example, configuring Servet filters that restrict the visibility of endpoints by some prefix/suffix, restrict the number of data records that can be evaluated in a time period, etc.

##### Deployment

Adding the wine color model:

```
$ curl -X PUT --data-binary @/path/to/wine-color.pmml -H "Content-type: text/xml" http://localhost:8080/openscoring/model/wine-color
```

The response body is an [`org.openscoring.common.ModelResponse`](https://github.com/openscoring/openscoring/blob/master/openscoring-common/src/main/java/org/openscoring/common/ModelResponse.java) object:

``` json
{
  "id" : "wine-color",
  "miningFunction" : "classification",
  "summary" : "Tree model",
  "properties" : {
    "created.timestamp" : "2016-06-19T21:35:58.592+0000",
    "accessed.timestamp" : null,
    "file.size" : 13537,
    "file.md5sum" : "1a4eb6324dc14c00188aeac2dfd6bb03"
  },
  "schema" : {
    "activeFields" : [ {
      "id" : "fixed_acidity",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "volatile_acidity",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "citric_acid",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "chlorides",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "total_sulfur_dioxide",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "density",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "pH",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "sulphates",
      "dataType" : "double",
      "opType" : "continuous"
    } ],
    "targetFields" : [ {
      "id" : "color",
      "dataType" : "string",
      "opType" : "categorical",
      "values" : [ "white", "red" ]
    } ],
    "outputFields" : [ {
      "id" : "probability_white",
      "dataType" : "double",
      "opType" : "continuous"
    }, {
      "id" : "probability_red",
      "dataType" : "double",
      "opType" : "continuous"
    } ]
  }
}
```

The pattern is to move all model-related logic to the server side, so that Openscoring client applications could be developed and used on a wide variety of platforms by people with varying degrees of experience.

All agents should be able to parse the above object at the basic model identification and schema level.
For example, understanding that the REST endpoint `/model/wine-color` holds a classification-type decision tree model, which consumes an eight-element input data record, and produces a three-element output data record. 

More sophisticated agents could rise to elevated model verification and field schema levels.
For example, checking that the reported file size and MD5 checksum are correct, and establishing field mappings between the model and the data store.

##### Evaluation

Evaluating the wine color model in single prediction mode:

```
$ curl -X POST --data-binary @/path/to/data_record.json -H "Content-type: application/json" http://localhost:8080/openscoring/model/wine-color
```

The request body is an [`org.openscoring.common.EvaluationRequest`](https://github.com/openscoring/openscoring/blob/master/openscoring-common/src/main/java/org/openscoring/common/EvaluationRequest.java) object:

``` json
{
  "id" : "sample-1",
  "arguments" : {
    "fixed_acidity" : 7.4,
    "volatile_acidity" : 0.7,
    "citric_acid" : 0,
    "chlorides" : 0.076,
    "total_sulfur_dioxide" : 34,
    "density" : 0.9978,
    "pH" : 3.51,
    "sulphates" : 0.56
  }
}
```

The response body is an [`org.openscoring.common.EvaluationResponse`](https://github.com/openscoring/openscoring/blob/master/openscoring-common/src/main/java/org/openscoring/common/EvaluationResponse.java) object:

``` json
{
  "id" : "sample-1",
  "result" : {
    "color" : "red",
    "probability_white" : 8.264462809917355E-4,
    "probability_red" : 0.9991735537190083
  }
}
```

Evaluating the wine color model in CSV mode:

```
$ curl -X POST --data-binary @/path/to/wine.csv -H "Content-type: text/plain; charset=UTF-8" http://localhost:8080/openscoring/model/wine-color/csv > /path/to/wine-color.csv
```

##### Undeployment

Removing the wine color model:

```
$ curl -X DELETE http://localhost:8080/openscoring/model/wine-color
```

##### Openscoring client libraries

The Openscoring REST API is fairly mature and stable.
The majority of changes happen in the "REST over HTTP(S)" transport layer. For example, adding support for new data formats and encodings, new user authentication mechanisms, etc.

Openscoring client libraries provide easy and effective means for keeping up with changes. Application developers get to focus on high-level routines such as "deploy", "evaluate" and "undeploy" commands, whose syntactics and semantics should remain stable for extended period of time.

The Java client library is part of the Openscoring project. Other client libraries (Python, R, PHP) are isolated into their own projects.

For example, the following Python script uses the [Openscoring-Python](https://github.com/openscoring/openscoring-python) library to replicate the example workflow.

``` python
import openscoring

os = openscoring.Openscoring("http://localhost:8080/openscoring")

# Deployment
os.deploy("wine-color", "/path/to/wine-color.pmml")

# Evaluation in single prediction mode
arguments = {
  "fixed_acidity" : 7.4,
  "volatile_acidity" : 0.7,
  "citric_acid" : 0,
  "chlorides" : 0.076,
  "total_sulfur_dioxide" : 34,
  "density" : 0.9978,
  "pH" : 3.51,
  "sulphates" : 0.56
}
result = os.evaluate("wine-color", arguments)
print(result)

# Evaluation in CSV mode
os.evaluateCsv("wine-color", "/path/to/wine.csv", "/path/to/wine-color.csv")

# Undeployment
os.undeploy("wine-color")
```
