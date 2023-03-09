---
layout: post
title: "Deploying LightGBM models on Java/JVM platform"
author: vruusmann
keywords: scikit-learn lightgbm sklearn2pmml jpmml-evaluator jpmml-transpiler builder-pattern
---

[LightGBM](https://github.com/Microsoft/LightGBM) is a gradient boosting framework that is written in the C++ language.

Most data scientists interact with LightGBM core APIs via high-level languages and APIs.
For example, Python users can choose between a medium-level [Training API](https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api) and a high-level [Scikit-Learn API](https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api) to meet their model training and deployment needs.

The lack of Java language bindings is understandable due to Java's subdued role in the ML space.
The [suggested route](https://github.com/microsoft/LightGBM/issues/909) is to suck it up and work with the low-level C++ library via the Java Native Interface (JNI).

This blog post details an alternative route for deploying LightGBM models on the Java/JVM platform:

1. Training a model using Scikit-Learn API.
2. Converting the model to the standardized PMML representation.
3. Deploying the model in "PMML interpretation" and "PMML to Java bytecode transpilation" modes.

## Model training

LightGBM has built-in support for categorical features and missing values.
This functionality often remains unused, because end users simply do not know about it, or cannot find a way to implement it in practice.
For example, Scikit-Learn version 0.22 still does not provide [missing value aware label encoders](https://github.com/scikit-learn/scikit-learn/pull/15009).

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package version 0.50.1 introduced utility functions `sklearn2pmml.preprocessing.lightgbm.make_lightgbm_dataframe_mapper` and `make_lightgbm_column_transformer` that take care of constructing column mapper transformations for complex datasets.

These two utility functions have identical signatures.
They accept the description of a dataset in the form of `dtypes` (iterable of `(column, dtype)` tuples) and `missing_value_aware` (boolean) parameters, and return a tuple `(mapper, categorical_feature)`.

The `sklearn_pandas.DataFrameMapper` meta-transformer is slightly more compact and computationally efficient than the `sklearn.compose.ColumnTransformer` meta-transformer. However, the former can only be applied to datasets where the column are referentiable by name (eg. `pandas.DataFrame`), whereas the latter can be applied to almost anything.

Training a model on a sparse mixed data type dataset:

``` python
from lightgbm import LGBMClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.lightgbm import make_lightgbm_dataframe_mapper

import pandas

df = pandas.read_csv("audit-NA.csv", na_values = ["N/A"])

columns = df.columns.tolist()

df_X = df[columns[: -1]]
df_y = df[columns[-1]]

# Drop boolean features
df_X = df_X.drop(["Deductions"], axis = 1)

mapper, categorical_feature = make_lightgbm_dataframe_mapper(df_X.dtypes, missing_value_aware = True)
classifier = LGBMClassifier(random_state = 13)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", classifier)
])
pipeline.fit(df_X, df_y, classifier__categorical_feature = categorical_feature)
```

The operational type (categorical vs. continuous) of each column is determined by its data type.
It is possible to force any column to become categorical by casting its data type to `pandas.api.types.CategoricalDtype`:

``` python
df_X["Age"] = df_X["Age"].astype("category")
```

The indices of categorical features (in the mapper output) must be passed to the `LGBMModel.fit(X, y, **fit_params)` method as the `categorical_feature` fit parameter.
By Scikit-Learn conventions, if the fit method is called via the `(PMML)Pipeline.fit(X, y, **fit_params)` method, then fit parameters need to be prefixed with the name of the step followed by two underscore characters.

## Model conversion

The PMML representation of LightGBM models relies on PMML markup and idioms that have been firmly in place since the PMML schema version 3.0.
This coupled with the fact that the PMML standard is designed with backward compatibility in mind makes for a very convincing argument that LightGBM models that have been converted to the PMML representation shall be usable for years and even decades with zero or very little maintenance.

A fitted `PMMLPipeline` object can be converted to a PMML XML file using the `sklearn2pmml.sklearn2pmml` utility function:

``` python
from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "LightGBMAudit.pmml")
```

The PMML converter collects and combines information from Scikit-Learn front-end (feature names, data types, transformations) and LighGBM back-end (the mathematical relationship between the features and the label).
The resulting "big picture" view is then analyzed, simplified and compacted.

A PMML XML file can be further transpiled (ie. translated + compiled) to a PMML service provider JAR file using the [JPMML-Transpiler](https://github.com/jpmml/jpmml-transpiler) command-line application:

```
$ java -jar jpmml-transpiler-executable-1.0-SNAPSHOT.jar --xml-input LightGBMAudit.pmml --jar-output LightGBMAudit.pmml.jar
```

A PMML service provider JAR file contains a single `org.dmg.pmml.PMML` subclass (source plus bytecode) that can be located and instantiated using [Java's service-provider loading facility](https://docs.oracle.com/javase/8/docs/api/java/util/ServiceLoader.html).

Transpilation has two major benefits to it.
First, all the object construction and initialization logic is embedded into class bytecode.
There is no need to ship around XML parsing and binding libraries, which simplifies deployment on limited or restricted runtime environments.
Second, dummy XML-backed elements are replaced with smart and optimized Java-backed elements.
The improvement in performance numbers depends on the model type and complexity.
At the time of writing this (December 2019), [LightGBM models should see around 15(!) times speed-up from transpilation](https://github.com/jpmml/jpmml-transpiler#benchmarking).

Nothing good comes without sacrifice.
Replacing static data structures with dynamic code leads to coupling with fairly narrow range of JPMML-Model and JPMML-Evaluator library versions.

The recommended approach is to use PMML XML files for long-term storage, and perform the transpilation to PMML service provider JAR files at the deployment time.

## Model deployment

The model should be evaluatable using any moderately capable PMML engine.
Java users are advised to head straight to the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

The central piece of the JPMML-Evaluator library is the `org.jpmml.evaluator.Evaluator` interface.

Obtaining an `Evaluator` object for a PMML XML file:

``` java
import org.jpmml.evaluator.LoadingModelEvaluatorBuilder;
import org.jpmml.evaluator.visitors.DefaultVisitorBattery;

File pmmlFile = new File("LightGBMAudit.pmml");

Evaluator result = new LoadingModelEvaluatorBuilder()
  // Ignore SAX Locator information to reduce memory consumption
  .setLocatable(false)
  // Pre-parse and intern PMML markup to improve performance and reduce memory consumption
  .setVisitors(new DefaultVisitorBattery())
  .load(pmmlFile)
  .build();
```

Obtaining an `Evaluator` object for a PMML service provider JAR file:

``` java
import org.jpmml.evaluator.ServiceLoadingModelEvaluatorBuilder;

File pmmlJarFile = new File("LightGBMAudit.pmml.jar");

URL pmmlJarURL = (pmmlJarFile.toURI()).toURL();

Evaluator evaluator = new ServiceLoadingModelEvaluatorBuilder()
  .loadService(pmmlJarURL)
  .build();
```

Querying and displaying the model schema:

``` java
System.out.println("Input (aka feature) fields: " + evaluator.getInputFields());
System.out.println("Primary result (aka target) fields: " + evaluator.getTargetFields());
System.out.println("Secondary result (aka output) fields: " + evaluator.getOutputFields());
```

The print-out displays eight input fields, one target field ("Adjusted") and two output fields ("probability(0)" and "probability(1)").
All fields are expressed in terms of the training dataset.
For example, the "Employment" input field is defined as a categorical string, whose valid value space contains seven elements (the PMML converter has completely reversed the effect of `LabelEncoder` or `PMMLLabelEncoder` transformers that are necessary for feeding string values into Scikit-Learn estimators).

Evaluating the model:

``` java
Map<FieldName, Object> arguments = new HashMap<>();
arguments.put(FieldName.create("Age"), 38);
arguments.put(FieldName.create("Employment"), "Private");
arguments.put(FieldName.create("Education"), "College");
arguments.put(FieldName.create("Marital"), "Unmarried");
arguments.put(FieldName.create("Occupation"), "Service");
arguments.put(FieldName.create("Income"), 81838);
arguments.put(FieldName.create("Gender"), null);
arguments.put(FieldName.create("Hours"), 72);

Map<FieldName, ?> results = evaluator.evaluate(arguments);

System.out.println(results);
```

The `Evaluator#evaluate(Map<FieldName, ?>)` method accepts a `Map` of arguments and returns another `Map` of results.
The ordering of map entries is not significant, because fields are identified by name not by position.
A missing value can be represented either by mapping the field name to a `null` reference, or by omitting the corresponding map entry altogether.

For example, evaluating the model with empty arguments:

``` java
Map<FieldName, Object> arguments = Collections.emptyMap();

Map<FieldName, ?> results = evaluator.evaluate(arguments);

System.out.println(results);
```

The prediction is a map of three entries `{Adjusted=0, probability(0)=0.9138749499794758, probability(1)=0.08612505002052413}`.
This can be regarded as the "baseline prediction" for this particular model configuration - "in the absence of all input, predict that there is a 91.4% probability of the event not happening".

The PMML approach to model API problematics is much more robust than Scikit-Learn or LightGBM approaches.
Most PMML documents can be verified and deployed without any external supporting documentation.
It is virtually impossible to make programming mistakes (eg. accidentally swapping the order of input fields), and the integration work does not need to be touched when updating model versions or upgrading from one model type to another.

## Resources

* "Audit-NA" dataset: [`audit-NA.csv`]({{ "/resources/data/audit-NA.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2019-12-03/train.py" | absolute_url }})