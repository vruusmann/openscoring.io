---
layout: post
title: "Extending Scikit-Learn with CHAID models"
author: vruusmann
keywords: CHAID jpmml-evaluator scikit-learn sklearn2pmml data-categorical data-missing
---

Scikit-Learn uses the Classification and Regression Trees (CART) algorithm for growing its decision trees.
This choice is driven by high conceptual compatibility between the two (preference towards dense numeric datasets), ease of implementation, and good performance characteristics.

The CART algorithm is so ingrained into the current code and data structures of the [`sklearn.tree`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree) module, that adding support for alternative algorithms has effectively been ruled out of the project roadmap.

## CART limitations ##

Scikit-Learn decision trees suffer from several functional issues:

* Limited support for categorical features.
All complex features must be transformed into simplified numerical features.
The default one-hot encoding transformation is suitable to low-cardinality categorical features, where category levels are independent of one another. Even the most sophisticated transformations (such as the ones provided by the [`category_encoders`](https://github.com/scikit-learn-contrib/category_encoders) package) are qualitatively inferior to native categorical feature support, because every helper transformation loses or distorts information in some way.
* No support for missing values.
All missing values must be replaced via imputation.
The effectiveness of the imputation transformation decreases as the sparsity of the dataset increases. The relevance of mean or mode values is questionable if the feature is more than 20% sparse, or if it does not adhere to the normal distribution function.
* No support for multi-way splits.
The only available split function is comparison against a threshold value, which splits the dataset into two subsets - data records that are "less than or equal" go to the "left" subset, whereas data records that are "greater than" go to the "right" subset.
A multi-way split can be emulated by a hierarchy of binary splits, but this quickly adds depth to decision trees, which is undesirable.

The CART algorithm is therefore rather sensitive towards the composition of the dataset.
It performs best with dense continuous features, and worst with sparse categorical features. The worst imaginable candidate would be a sparse high-cardinality categorical feature, where category levels are not independent of one another, but aggregate into a number of clusters (eg. USPS ZIP Code).

If the dataset is rich in complex features, then it will be smart to explore alternative algorithms such as CHAID, C4.5 or C5.0.

## CHAID implementation ##

At the time of writing this (July 2022), there are no suitable Scikit-Learn extension packages available.
The workaround is to choose a Python-based algorithm package, and then integrate it with Scikit-Learn by ourselves.

Chi-Squared Automatic Inference Detection (CHAID) is one of the oldest algorithms, but is perfectly fine for use in the most demanding applications.
Its success and versatility hinges on the fact that it operates with categorical (aka nominal) features. If a dataset contains continuous features, then they need to be analyzed and discretized externally.

Arguably, **it is much easier to discretize continuous features (ie. CHAID requirement) than to normalize categorical features (ie. CART requirement)**.
Discretization methods range from simple uni- and multivariate statistics to complex optimizers such as the [`optbinning`](https://github.com/guillermo-navas-palencia/optbinning) package.

A single continuous feature may be discretized using different discretization methods and parameterizations.
One end of the spectrum is binning into two classes (aka thresholding). The other end is performing an operational type cast from continuous to categorical, so that the number of bins equals the number of unique feature values in the training dataset.

Missing values are treated as a special category level.
During splitting, they can form a standalone branch, or be aggregated with other category levels.

Decision tree algorithms do not care if the training dataset contains multiple collinear features, so there is no practical reason to withhold from experimenting with multiple competing feature transformations.
If two or more features appear collinear for some subset (ie. not collinear for the training dataset as a whole, but collinear for some segment of it), then the winner is picked randomly.

The [`CHAID`](https://github.com/Rambatino/CHAID) package provides the `CHAID.Tree` class, which codifies a complete and highly parameterizable CHAID algorithm implementation.
However, its API is rather uncommon and lacks some key interactions points.

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package version 0.84 provides `sklearn2pmml.tree.chaid.CHAIDClassifier` and `sklearn2pmml.tree.chaid.CHAIDRegressor` models, which make the `CHAID.Tree` class embeddable into Scikit-Learn pipelines, and commandable via familiar `fit(X, y)` and `predict(X)` methods.

## Training ##

The `CHAIDEstimator.fit(X, y)` method assumes that all columns of the `X` dataset are categorical features.
If the `X` dataset contains continuous features (eg. a `float` or `double` column, with many distinct values) then they shall be automatically casted to categorical features.

However, it is advisable to make this conversion explicit by passing all features through the `sklearn2pmml.decoration.CategoricalDomain` decorator:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain

def make_passthrough_mapper(cols):
  return DataFrameMapper(
    [([col], CategoricalDomain()) for col in cols]
  )
```

For example, it is possible to apply `CHAIDClassifier` directly to the raw "iris" dataset:

``` python
from sklearn.datasets import load_iris
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.tree.chaid import CHAIDClassifier

iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)

def make_classifier(max_depth = 5):
  config = {
    "max_depth" : max_depth
  }
  return CHAIDClassifier(config = config)

pipeline = PMMLPipeline([
  ("mapper", make_passthrough_mapper()),
  ("classifier", make_classifier(max_depth = 3))
])
pipeline.fit(iris_X, iris_y)

sklearn2pmml(pipeline, "CHAIDIris.pmml")
```

The eventual type definitions of fields are easy to verify by opening the PMML document in a text editor, and inspecting `DataField@optype` and `DataType@dataType` attribute values.

For example, the `/PMML/DataDictionary` element for the "CHAIDIris" model contains three field declarations - one for the target field, and two for input fields:

``` xml
<DataDictionary>
  <DataField name="target" optype="categorical" dataType="integer">
    <Value value="0"/>
    <Value value="1"/>
    <Value value="2"/>
  </DataField>
  <DataField name="sepal width (cm)" optype="categorical" dataType="double">
    <Value value="2.0"/>
    <Value value="2.2"/>
    <Value value="2.3"/>
    <!-- omitted 19 intermediate Value elements -->
    <Value value="4.4"/>
  </DataField>
    <DataField name="petal length (cm)" optype="categorical" dataType="double">
    <Value value="1.0"/>
    <Value value="1.1"/>
    <Value value="1.2"/>
    <!-- omitted 39 intermediate Value elements -->
    <Value value="6.9"/>
  </DataField>
</DataDictionary>
```

The category levels that are listed under a `DataField` element fully define the valid value space for that field.
Any attempt to evaluate a model with an unlisted input value shall fail with a value error.

If the goal is to preserve the "normalcy" of continuous features, and even allow for some interpolation and extrapolation, then they should be binned into (temporary-) categorical features.
Scikit-Learn provides the [`slearn.preprocessing.KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) transformer, which can learn bin edges based on common statistical procedures.

No matter which binning strategy is chosen, the results should be encoded using the ordinal encoding method.
The reason for this is maintaining feature's integrity.
Ordinal encoding yields a single categorical integer feature, whereas one-hot encodings yield many binary indicator features. The CHAID algorithm cannot (re-)aggregate binary indicators into larger chunks based on their parent, and would therefore resort to generating CART-style binary splits (signalling if a particular category level is "on" or "off"), instead of proper CHAID-style multy-way splits.

``` python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

def make_mapper(cat_cols, cont_cols):
  return DataFrameMapper(
    [([cat_col], CategoricalDomain()) for cat_col in cat_cols] +
    [(cont_cols, [ContinuousDomain(), KBinsDiscretizer(n_bins = 5, encode = "ordinal", strategy = "quantile")])]
  )
```

The `KBinsDiscretizer` transformer is currently unable to deal with missing values.
The suggested workaround is to emulate its statistical procedure(s) in a missing value-aware way, and then use the learned bin edges for parameterizing a general-purpose `sklearn2pmml.preprocessing.CutTransformer` transformer:

``` python
def make_sparse_mapper(df, cat_cols, cont_cols):
  binners = dict()
  for cont_col in cont_cols:
    bins = numpy.nanquantile(df[cont_col], q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    # Deduplicate and convert from Numpy scalar float to Python float
    bins = [float(bin) for bin in dict.fromkeys(bins)]
    labels = list(range(0, len(bins) - 1))
    binners[cont_col] = CutTransformer(bins = bins, labels = labels)

  return DataFrameMapper(
    [([cat_col], CategoricalDomain()) for cat_col in cat_cols] +
    [([cont_col], [ContinuousDomain(), binners[cont_col]]) for cont_col in cont_cols]
  )
```

## Prediction ##

The `CHAID.Tree` class is a data exploration and mining tool. It does not provide any Python API for making predictions on new datasets (see [Issue 128](https://github.com/Rambatino/CHAID/issues/128)).

Instead of extending the `sklearn2pmml.tree.chaid.CHAIDEstimator` base class with custom prediction methods, it is much easier to export the core decision tree data structure in Predictive Model Markup Language (PMML) data format, and make predictions using a PMML engine instead.

Conversion to PMML:

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
  ("mapper", make_mapper(...)),
  ("estimator", make_chaid_estimator(...))
])
pipeline.fit(X, y)

sklearn2pmml(pipeline, "CHAID.pmml")
```

When it comes to PMML engines, then the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library is miles ahead of competition in terms of maturity, quality of design and engineering, and plain performance figures. The core library is written in Java, but there are easy integrations available for most common ML frameworks.

Python users can install and use JPMML-Evaluator as the [`jpmml_evaluator`](https://github.com/jpmml/jpmml-evaluator-python) package.

The JPMML-Evaluator API distinguises itself from Scikit-Learn conventions by calling its main entry point method `Evaluator.evaluate(X)`.
This distinction is necessary to communicate that JPMML-Evaluator uses a simpler and more economical "business logic", where all prediction aspects are computed and exposed simultaneously.

A decision tree classifier has three main prediction aspects:

1. `predict(X)` - The class of the winning node.
2. `predict_proba(X)` - The class probability distribution of the winning node.
3. `apply(X)` - The index of the winning node.

The `Evaluator.evaluate(X)` method accepts a `pandas.DataFrame` object, and returns a new `pandas.DataFrame` object, which packs all three prediction aspects.

For maximum portability, PMML input and result fields are identified by name, not by position.
The complete information (name, type, value space, etc.) about individual fields can be queried from the underlying model schema using `Evaluator.getInputFields()` and `Evaluator.getTargetFields()` methods.

The fact that PMML is a strongly typed language can be used for performing additional data sanity and consistency checks when crossing important boundaries.

The case in point is the encoding of missing values. Python APIs are typically pretty lax around here. For example, a dataset may have an `object` data type column, which contains a mix of string, `None` and `NaN` values:

``` python
import numpy

X = numpy.asarray(["puppy", None, float("NaN")])
print(X.dtype)
print(X)
```

This kind of morphism (type and value auto-conversions) is beyond PMML tolerance.

Specifically, PMML uses a system where values are assigned to valid, invalid and missing value spaces for data validation and pre-processing purposes.
The `NaN` value is assigned to the invalid value space (ie. a reserved floating-point constant that "short-circuits" arithmetic operations at software and/or hardware layers).
A model shall reject any evaluation requests that contain invalid values by throwing an `org.jpmml.evaluator.InvalidResultException` with a message like `Field <name> cannot accept user input value "NaN"`.

Data validation errors could be solved by editing model files and activating value replacements and/or value space conversions for the affected fields.
However, it would be more sustainable to deal with the real source of problems, which is about Python's bad habit of using the `NaN` value to denote missing values even in non-numeric columns.

Canonicalizing a dataset by replacing all `NaN` values with `None` values:

``` python
import numpy
import pandas

df = pandas.read_csv(...)
df = df.replace({numpy.nan: None})
```

The PMML representation of Scikit-Learn pipelines is completely free from any Scikit-Learn extension package dependencies.
The deployment environment only needs to declare a `jpmml_evaluator` dependency. It is advisable to always be using the latest version, as newer versions deliver more and improved Python-to-Java connectivity options, and overall better user experience.

Loading and evaluating a model:

``` python
from jpmml_evaluator import make_evaluator

import pandas

# Load model
evaluator = make_evaluator("CHAID.pmml") \
  .verify()

# Load and canonicalize arguments
df = pandas.read_csv("input.csv", sep = "\t", na_values = ["N/A", "NA"])
df = df.replace({numpy.nan: None})

# Evaluate
df_pred = evaluator.evaluateAll(df)

# Save results
df_pred.to_csv("output.csv", sep = "\t", index = False)
```

## Resources ##

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* "Audit-NA" dataset: [`audit-NA.csv`]({{ "/resources/data/audit-NA.csv" | absolute_url }})
* Python scripts: [`train-iris.py`]({{ "/resources/2022-07-14/train-iris.py" | absolute_url }}), [`train-audit.py`]({{ "/resources/2022-07-14/train-audit.py" | absolute_url }}) and [`predict.py`]({{ "/resources/2022-07-14/predict.py" | absolute_url }})
