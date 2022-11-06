---
layout: post
title: "Converting logistic regression models to PMML documents"
author: vruusmann
keywords: scikit-learn r apache-spark pyspark sklearn2pmml r2pmml pyspark2pmml
---

Logistic regression is often the go-to algorithm for binary classification problems.

This blog post demonstrates how to perform feature engineering and train a logistic regression model in a way that allows for quick productionization using the Predective Model Markup Language (PMML) standard.
The same workflow is implemented using Scikit-Learn, R and Apache Spark frameworks to demostrate their particularities.

Summary of the workflow:

* Ingesting the raw dataset.
* Feature engineering:
  * Capturing and refining feature information.
  * Applying transformations to individual continuous and categorical features.
  * Interacting categorical features.
* Training a model using the transformed dataset.
* Enhancing the model with verification data.
* Converting the model to a PMML document using JPMML family conversion tools and libraries.

### Scikit-Learn

Scikit-Learn follows object-oriented programming (OOP) paradigm.
The [Linear Models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) module provides a `LinearModel` base class, which is subclassed and mixed with `RegressorMixin` and `ClassifierMixin` traits to provide algorithm-specific model base classes.
The logistic regression algorithm is available as the [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model.

Scikit-Learn estimators are trained by calling the `fit(X, y, **fit_params)` method.
However, real-life datasets require serious data pre-processing before they can be passed to this method.
The main requirement is transforming features from the mixed high-level representation to the unified (floating point) low-level representation so that they would become "understandable" to numerical algorithms.

Scikit-Learn provides a decent selection of transformers that help with importing data into the pipeline. Unfortunately, the situation is rather bleak when it comes to manipulating or modifying data inside the pipeline (eg. concatenating two string columns into a new string column).

The `sklearn2pmml` package does its best to address this deficiency in a PMML compatible manner.
The [Decoration](https://github.com/jpmml/sklearn2pmml/tree/master/sklearn2pmml/decoration) and [Preprocessing](https://github.com/jpmml/sklearn2pmml/tree/master/sklearn2pmml/preprocessing) modules provide transformers for performing common data science operations.
They operate on the high-level representation of data, and typically precede any Scikit-Learn transformers in the pipeline.

Transforming the "audit" dataset to a 2-D Numpy array:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import ExpressionTransformer, LookupTransformer

import numpy

employment_mapping = {
  "Consultant" : "Private",
  "Private" : "Private",
  "PSFederal" : "Public",
  "PSLocal" : "Public",
  "PSState" : "Public",
  "SelfEmp" : "Private",
  "Volunteer" : "Other"
}

mapper = DataFrameMapper([
  (["Income"], [ContinuousDomain(), ExpressionTransformer("numpy.log(X[0])", dtype = numpy.float64)]),
  (["Employment"], [CategoricalDomain(), LookupTransformer(employment_mapping, default_value = None), OneHotEncoder(drop = "first")]),
  (["Gender", "Marital"], [MultiDomain([CategoricalDomain(), CategoricalDomain()]), OneHotEncoder(), PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)]),
  (["Age", "Hours"], [ContinuousDomain(), StandardScaler()]), 
  ("Education", [CategoricalDomain(), OneHotEncoder(drop = "first")]),
  ("Occupation", [CategoricalDomain(), OneHotEncoder(drop = "first")])
])

df_Xt = mapper.fit_transform(df_X)
```

There are several options for converting strings to bit vectors:

| Transformer | Scikit-Learn version | Input arity | Quick drop category? | Output |
|-------------|----------------------|-------------|----------------------|--------|
| `LabelBinarizer` | All | 1 | No | Dense array |
| `[LabelEncoder(), OneHotEncoder()]` | < 0.20 | 1 | No | Sparse matrix |
| `OneHotEncoder()` | >= 0.20 | 1 or more | Yes | Sparse matrix |

The [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) transformer was heavily refactored in Scikit-Learn version 0.20, giving it the ability to fit and transform multiple columns together, and drop category levels.

These two abilities enable vastly cleaner and conciser workflows.

The interaction between "Gender" and "Marital" string columns can be expressed as a one-liner.
The list selector syntax (`["Gender", "Marital"]`) yields a two-column string array, which is first one-hot-encoded to an eight-column integer array, and then polynomially combined into a 36-column integer array.
The first eight elements correspond to raw category levels (ie. `Gender=Male, Gender=Female, Marital=Absent, ..`), and the remaining twenty eight ((8 * (8 - 1)) / 2) elements to interactions between them (ie. `Gender=Male * Gender=Female, Gender=Male * Marital=Absent, ..`).

Using the [`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) transformer for feature interactions does the job, but is far from elegance and efficiency.
The main complaint is that it lacks the concept of feature boundaries ("treat the leading n elements as belonging to feature A, and the following m elements as belonging to feature B"), which could be used to prevent the generation of meaningless or undesirable interaction terms.
For example, interaction terms which combine different category levels of the same feature (eg. `Gender=Male * Gender=Female`) are non-sensical from the real-life perspective, and risk blowing up numerical algorithms due to high collinearity with other terms.

Fighting collinearity is a major issue when training unregularized (logistic-) regression models.
A common source of highly correlated features is the binarization or one-hot-encoding of string columns.
The `OneHotEncoder` transformer fixes this by allowing one category level to be excluded from the one-hot-encoding process.
Most data scientist habitually drop the first category level.

The logistic regression model is associated with transformations by constructing a two-step pipeline.
The `PMMLPipeline` object is enhanced with verification data and converted to the PMML representation using the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package:

``` python
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", LogisticRegression(multi_class = "ovr", max_iter = 1000))
])
pipeline.fit(df_X, df_y)

pipeline.verify(df_X.sample(n = 10))

sklearn2pmml(pipeline, "SkLearnAudit.pmml")
```

### R

R follows functional programming paradigm.
The built-in `stats` package provides a [`glm()`](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/glm) function for training generalized linear models.
The logistic regression mode is activated by setting the `family` argument to binomial value (either as a string literal or a [`family`](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family) object).

If the goal is to perform feature engineering in a PMML compatible manner, then the `glm()` function must be called using "formula interface".
Simple formulas can be specified inline (eg. `glm(Adjusted ~ ., family = "binomial", data = audit.df)`).
Complex formulas should be assembled step by step from stringified terms, and then compiled into a standalone [`formula`](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/formula) object:

``` r
audit.terms = c("Adjusted ~ .")

# Feature engineering
audit.terms = c(audit.terms, ...)

audit.formula = as.formula(paste(audit.terms, collapse = " "))

audit.glm = glm(formula = audit.formula, family = binomial(link = "logit"), data = audit.df)
```

Feature engineering is possible by embedding R function calls into model formulae.
The portability of model formulae can be improved by using fully-qualified function names (ie. `<namespace>::<name>()` instead of `<name>()`).

If the model will be used only in R environment, then it is possible to use any R language operator or function.
However, if the model needs to be converted to the PMML representation, then it is possible to use only those constructs that are recognized and supported by the conversion tool.

Continuous features can be manipulated using arithmetic operators and functions:

``` r
# ?I
audit.terms = c(audit.terms, "+ I(log(Income)) - Income")
```

Most R's algorithms detect column data types, and treat continuous (eg. `numeric` and `integer`) and categorical (eg. `logical`, `factor`) features using different subroutines.

The `glm()` function automatically drops the first category level of each categorical feature to fight collinearity.
For example, a boolean feature gives rise to exactly one categorical predictor term (typically the `true` category, because `factor` levels follow the natural ordering by default).

Categorical features can be regrouped using `plyr::revalue()` or `plyr::mapvalues()` functions.
All arguments to the function call must be formatted as strings so that they could become an integral part of the `formula` object:

``` r
# Declare the replacement table as a named vector
employment.newlevels = c(
  "Consultant" = "Private",
  "Private" = "Private",
  "PSFederal" = "Public",
  "PSLocal" = "Public",
  "PSState" = "Public",
  "SelfEmp" = "Private",
  "Volunteer" = "Other"
)

# Format the named vector as a string
# Escape both names and values using the `shQuote()` function
employment.newlevels.str = paste("c(", paste(lapply(names(employment.newlevels), function(x){ paste(shQuote(x), "=", shQuote(employment.newlevels[[x]])) }), collapse = ", "), ")", sep = "")
print(employment.newlevels.str)

# ?plyr::revalue
audit.terms = c(audit.terms, paste("+ plyr::revalue(Employment, replace = ", employment.newlevels.str, ") - Employment", sep = ""))
```

Feature interactions (between all feature types) can declared using the `:` operator:

``` r
# ?interaction()
audit.terms = c(audit.terms, "+ Gender:Marital")
```

The logistic regression model together with the embedded formula is converted to the PMML representation using the [`r2pmml`](https://github.com/jpmml/r2pmml) package.
The legacy `pmml` package supports model formulae only partially, and should be avoided.

Right before the conversion, the logistic regression model object is enhanced with verification data using the `r2pmml::verify.glm()` function.
The `audit.glm` variable is re-assigned, because this function returns a modified copy of the input (rather than modifying the input in place).

``` r
library("dplyr")
library("r2pmml")

audit.glm = r2pmml::verify(audit.glm, newdata = dplyr::sample_n(audit.df, 10))

r2pmml::r2pmml(audit.glm, "RExpAudit.pmml")
```

### Apache Spark

Apache Spark allows the end user to choose between programming paradigms.
The prevailing DataFrame-based machine learning API called Apache Spark ML is built around transformers and models that are almost identical to their Scikit-Learn namesakes.
In fact, it is possible to translate pipelines between these two ML frameworks with not much effort.

However, solving data science problems at Apache Spark ML layer involves lot of typing, and sooner or later hits various API limits.
For example, the label column of classification models must be explicitly converted from the high-level string representation to low-level bit vector representation, and back, using a pair of `StringIndexer(Model)` and `IndexToString` transformers.
The initialization of each pipeline stage typically requires writing three to five lines of boilerplate code, which adds to the burden.

It is possible to "compress" rather complex workflows into small and expressive scripts by leveraging different API layers.
For example, performing all data extraction, transformation and loading (ETL) work in the Apache Spark SQL layer, and assembling the pipeline using R-like model formulae in the Apache Spark ML layer:

``` python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RFormula, SQLTransformer

sqlTransformer = SQLTransformer(statement = "SELECT * FROM __THIS__")
rFormula = RFormula(formula = "Adjusted ~ .")
classifier = LogisticRegression()

pipeline = Pipeline(stages = [sqlTransformer, rFormula, classifier])
```

Apache Spark SQL supports most standard SQL constructs and functions.
SQL and PMML are are conceptually rather close in their underlying data models (strongly typed, scalar values).
The [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library provides an `org.jpmml.sparkml.ExpressionTranslator` component for translating Catalyst expressions to PMML expressions.
This component currently supports around 45 expression types.
If some expression is not supported, then it is often possible to work around it by re-expressing it in terms of other supported functions.

Even though Scikit-Learn and Apache Spark both call their workflow unit a "pipeline", they are very different by design and implementation.

Scikit-Learn pipelines are linear sequences of steps (aka stages).
The dataset is a contiguous array or matrix that is automatically passed from one step to the next step. There can be no back-references to earlier steps or earlier states of the dataset (eg. "get the second column of the dataset three steps back from here").

In contrast, Apache Spark pipelines are directed acyclic graphs of stages, optimized for lazy and distributed evaluation.
The dataset is a loose collection of columns. Each stage pulls in a subset of existing columns (`HasInputCol` and `HasInputCols` traits) and pushes out new columns (`HasOutputCol` and `HasOutputCols` traits).
Created columns stay in place until replaced or removed.

Logistic regression models can be represented using two different PMML model elements.
The [`GeneralRegressionModel`](http://dmg.org/pmml/v4-4-1/GeneralRegression.html) element is more flexible (eg. contrast matrices, parameterizable link functions), but is encoded in a matrix-oriented way that is rather difficult to parse and follow for humans.
The [`RegressionModel`](http://dmg.org/pmml/v4-4-1/Regression.html) elements loses in functionality but makes it up in human-friendliness.

The JPMML-SparkML library allows the end user to choose between them by setting the value of the `org.jpmml.sparkml.model.HasRegressionTableOptions#OPTION_REPRESENTATION` conversion option to `GeneralRegressionModel` or `RegressionModel` string literals, respectively.

The pipeline model is enhanced with verification data and converted to the PMML representation using the [`pyspark2pmml`](https://github.com/jpmml/pyspark2pmml) package:

``` python
from pyspark2pmml import PMMLBuilder

pipelineModel = pipeline.fit(df)

pmmlBuilder = PMMLBuilder(sc, df, pipelineModel) \
  .putOption(classifier, "representation", "RegressionModel") \
  .verify(df.sample(False, 0.005))

pmmlBuilder.buildFile("PySparkAudit.pmml")
```

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* R script: [`train.R`]({{ "/resources/2020-01-19/train.R" | absolute_url }})
* Python scripts: [`train-sklearn.py`]({{ "/resources/2020-01-19/train-sklearn.py" | absolute_url }}) and [`train-pyspark.py`]({{ "/resources/2020-01-19/train-pyspark.py" | absolute_url }})