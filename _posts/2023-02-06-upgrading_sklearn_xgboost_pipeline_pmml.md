---
layout: post
title: "Upgrading Scikit-Learn XGBoost pipelines"
author: vruusmann
keywords: scikit-learn xgboost sklearn2pmml data-categorical
---

XGBoost is one of the top algorithms for solving Tabular ML problems.

The [`xgboost`](https://github.com/dmlc/xgboost/tree/master/python-package) package provides XGBoost functionality in two flavours.
First, the low-level [Python Learning API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training) aims at API parity with the underlying C(++) library. Second, the high-level [Scikit-Learn API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn) aims at making the most popular parts accessible via Scikit-Learn style wrappers.

According to the [XGBoost PyPI release history](https://pypi.org/project/xgboost/#history), the `xgboost` package has been publicly available since mid-2015.
Initial releases (0.4, 0.6) carry "pre-release" markers.
The first production-ready release is XGBoost version 0.7 (aka 0.70), which is dated 1st of January, 2018.

This blog post walks through all major XGBoost releases from 0.7 to 1.7, with the intent of pinpointing any Scikit-Learn API changes, and making observations and comparisons from the end user perspective.
The focus is on handling categorical data.
Any measurable changes in predictive or computational performance are ignored. The general expectation is that each new major XGBoost release version is smarter and faster than all the previous ones.

## XGBoost version 0.7 (v0.72.1) ##

The "audit" dataset deals with a binary classification problem.

The label column is originally a two-valued integer, but it is translated into a two-valued string in order to make the experiment more challenging.

There are nine feature columns, three continuous and six categorical.

``` python
import pandas

df = pandas.read_csv("audit.csv")
df["Adjusted"] = df["Adjusted"].apply(lambda x: ("yes" if x else "no"))

cat_cols = ["Deductions", "Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Income", "Hours"]

X = df[cat_cols + cont_cols]
y = df["Adjusted"]
```

A good Scikit-Learn pipeline should start with an "initializer" step that ensures that the incoming data matrix is correct (eg. right columns, in the right order).

Both `sklearn_pandas.DataFrameMapper` and `sklearn.compose.ColumnTransformer` (meta-)transformers offer combined data matrix canonicalization and transformation functionality.
This experiment goes with the `DataFrameMapper` class, because it has simpler and more intuitive API, is purpose-built for dealing with `pandas.DataFrame` inputs and outputs, and works with any Scikit-Learn version.

Constructing an OHE-style initializer:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

import numpy

def make_mapper():
  return DataFrameMapper(
    [([cont_col], [ContinuousDomain()]) for cont_col in cont_cols] +
    [([cat_col], [CategoricalDomain(), OneHotEncoder(sparse_output = False, dtype = numpy.int8)]) for cat_col in cat_cols]
  , input_df = True, df_out = True)
```

Numeric columns can be mapped as-is.
However, string and boolean columns must be transformed into numeric columns.

A data scientist is free to employ any third-party package, any string-to-numeric encoding algorithm here.
For example, the [`category_encoders`](https://github.com/scikit-learn-contrib/category_encoders) package provides a plethora of Scikit-Learn compatible transformers, which are often based on the state-of-the-art ML research publications.

The `scikit-learn` package is rather hapless in comparison.
In fact, the only option is the `sklearn.preprocessing.OneHotEncoder` transformer, which performs a deep, two-level encoding of categorical columns (ie. first from string to ordinal integer, and then from ordinal integer to a list of binary indicators).

The other candidate, the `sklearn.preprocessing.OrdinalEncoder` transformer fails in the vetting process, because it performs a shallow, one-level encoding (ie. from string to ordinal integer).
Unlike the LightGBM algorithm, the XGBoost algorithm is not programmed to handle such "partially" encoded columns.
If ordinal integers slip through, then they will be treated analogously to continuous integers, which qualifies as a serious mistake.

**Important**: A pipeline where Scikit-Learn's `OneHotEncoder` transformer is combined with an XGBoost estimator can only be applied to dense datasets, and never to sparse datasets.
In brief, the `OneHotEncoder` transformer is functionally restricted to two-state output.
This is sufficient for encoding dense datasets (ie. "on" and "off" states), but not sparse datasets (ie. "on", "off" and "unknown" states).

**Important**: A Scikit-Learn's `OneHotEncoder` transformer and an XGBoost estimator are not compatible with one another in their default configurations.
In brief, the `OneHotEncoder` transformer produces sparse data matrices by default, where the `1` value represents the "on" state and the omitted value represents the "off" state. However, an XGBoost estimator (mis-)interprets omitted values as the "unknown" state.

The above notes have been discussed in detail in an earlier blog post about [one-hot encoding categorical features in Scikit-Learn XGBoost pipelines]({% post_url 2022-04-12-onehot_encoding_sklearn_xgboost_pipeline %}).

There are two pipeline configurations that yield provably correct results:

* Making OHE output dense - `[OneHotEncoder(sparse_output = False, dtype = numpy.int8), XGBModel()]`.
* Keeping OHE output as sparse, and instructing XGBoost to interpret some value (other than `float("NaN")`) as missing value - `[OneHotEncoder(sparse_output = True), XGBModel(missing = -999)]`.

The first configuration is more robust than the second (easier to understand, less likely to break after a scheduled Scikit-Learn or XGBoost version update).
However, it involves a matrix densification operation, which can raise memory requirements by several orders of magnitude. When going this route, then the impact can be softened somewhat by changing the data type of output data matrix elements to some very small and cheap one such as `numpy.(u)int8` or `bool`.

Constructing an OHE-style XGBoost estimator:

``` python
from xgboost.sklearn import XGBClassifier

def make_classifier():
  return XGBClassifier(objective = "binary:logistic", n_estimators = 131, max_depth = 6)
```

The values of `XGBModel.n_estimators` and `XGBModel.max_depth` attributes are set to values that should let the XGBoost algorithm to run to exhaustion with this dataset.

Constructing and fitting a pipeline:

``` python
from sklearn.pipeline import Pipeline

mapper = make_mapper()
classifier = make_classifier()

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", classifier)
])
pipeline.fit(X, y)
``` 

The XGBoost classifier performs label encoding and decoding during `XGBClassifier.fit(X, y)` and `XGBClassifier.predict(X)` method calls similarly to Scikit-Learn classifiers.
The learned encoding is stored in the `XGBClassifier._le` attribute as a `sklearn.preprocessing.LabelEncoder` object.

Exporting the booster object:

``` python
classifier._Booster.save_model("Booster.bin")
```

The `XGBModel._Booster` attribute (not to be confused with the `XGBModel.booster` attribute!) holds a reference to the underlying `xgboost.Booster` object.

Saving the booster object into a file is the suggested way for moving XGBoost models between ML frameworks and applications.
Unfortunately, the only supported data format is a binary proprietary one, for which there is no public specification or documentation available.

When making pipeline configuration changes, then it is possible to monitor both its intended and unintended consequences by diffing booster files.
For example, when updating parameterizations, or replacing Scikit-Learn's `OneHotEncoder` transformer with some third-party implementation, then the byte content of the booster file must remain the same.

## XGBoost versions 0.8 and 0.9 (v0.82, v0.90) ##

Added the [`XGBModel.save_model(fname)`](https://github.com/dmlc/xgboost/blob/v0.82/python-package/xgboost/sklearn.py#L246-L262) method:

Exporting the booster object:

``` python
classifier.save_model("Booster.bin")
```

## XGBoost version 1.0 (v1.0.2) ##

Added the [`XGBoostLabelEncoder`](https://github.com/dmlc/xgboost/blob/v1.0.2/python-package/xgboost/compat.py#L140-L161) class, and changed the type of the `XGBClassifier._le` attribute to it.

Exporting the booster object is now also possible in JSON data format:

``` python
# Note the ".json" filename extension!
classifier.save_model("Booster.json")
```

XGBoost generates a "minified" JSON document, which does not contain any newlines, indentation or even intermittent whitespace.
Such a multi-megabyte, non-breaking line of text poses a serious challenge to most text editors.

Pretty-printing a booster JSON file to make it more human friendly:

```
$ python -m json.tool < Booster.json > Booster-pretty.json
``` 

## XGBoost versions 1.1 and 1.2 (v1.1.1, v1.2.1) ##

No changes.

## XGBoost version 1.3 (v1.3.3) ##

Added the [`XGBClassifier.use_label_encoder`](https://github.com/dmlc/xgboost/blob/v1.3.3/python-package/xgboost/sklearn.py#L814) attribute.

The `XGBClassifier.fit(X, y)` method emits a user warning that label encoding should be done manually:

```
/usr/local/lib/python3.9/site-packages/xgboost/sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
```

This request goes against Scikit-Learn conventions that label management is the responsibility of the ML framework.

Extracting the label encoder:

``` python
from pandas import Series
from xgboost.compat import XGBoostLabelEncoder

def make_classifier():
  return XGBClassifier(
    objective = "binary:logistic", n_estimators = 131, max_depth = 6,
    use_label_encoder = False
  )

classifier = make_classifier()

xgb_le = XGBoostLabelEncoder()
y = Series(xgb_le.fit_transform(y), name = "Adjusted")

classifier._le = xgb_le
```

The extracted label encoder is stored in the `XGBClassifier._le` attribute just like before.

It is clear from the XGBoost library source code that this attribute has retained its special status.
For example, the [`XGBClassifier.predict(X)`](https://github.com/dmlc/xgboost/blob/v1.3.3/python-package/xgboost/sklearn.py#L931-L987) method checks for its presence and, if found, automatically post-processes the raw prediction using its `inverse_transform(y)` method.

According to [XGBoost version 1.3 release notes](https://github.com/dmlc/xgboost/releases/tag/v1.3.0), the library now supports direct categorical splits.
This functionality is considered highly experimental. It is partially integrated into the low-level Python Learning API, but not into the high-level Scikit-Learn API.

A booster object that contains categorical splits cannot be saved in binary proprietary data format.

## XGBoost version 1.4 (v1.4.2) ##

The booster JSON file contains an embedded feature map (lists of feature names and feature types).

When making predictions, then the XGBoost library uses this information to ensure that the new validation or testing datasets are structurally similar to the training dataset.
Scikit-Learn pipelines should pass such checks cleanly thanks to the "initializer" step.

## XGBoost version 1.5 (v1.5.2) ##

Added the [`XGBModel.enable_categorical`](https://github.com/dmlc/xgboost/blob/v1.5.2/python-package/xgboost/sklearn.py#L401) attribute.

Starting from here, the support for direct categorical splits is available at all API levels.

Constructing a categorical-style initializer:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

def make_mapper():
  return DataFrameMapper(
    [([cont_col], [ContinuousDomain()]) for cont_col in cont_cols] +
    [([cat_col], [CategoricalDomain(dtype = "category")]) for cat_col in cat_cols]
  , input_df = True, df_out = True)
```

A categorical column can be marked as such by simply casting it to the [`pandas.CategoricalDtype`](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html) data type.

**Important**: A quick cast using the `category` data type alias is dataset-dependent.
If the same pipeline object is used for making predictions on different datasets, then the `dtype` argument must be a full-blown `CategoricalDtype` object, which enumerates all valid category levels ahead of time.

Removing Scikit-Learn's `OneHotEncoder` transformer lifts all restrictions that were previously imposed by it.
Specifically, the same pipeline can now be applied to both dense and sparse datasets.

Constructing a categorical-style XGBoost estimator:

``` python
from xgboost.sklearn import XGBClassifier

def make_classifier():
  return XGBClassifier(
    objective = "binary:logistic", n_estimators = 131, max_depth = 6,
    tree_method = "gpu_hist", enable_categorical = True,
    use_label_encoder = False
  )
```

The `enable_categorical` argument acts as a double confirmation mechanism.
If not explicitly set to `True`, then the fit method shall fail with the following value error:

```
Traceback (most recent call last):
  File "train-categorical.py", line 45, in <module>
    pipeline.fit(X, y)
  File "/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py", line 406, in fit
    self._final_estimator.fit(Xt, y, **fit_params_last_step)
  ...
  File "/usr/local/lib/python3.9/site-packages/xgboost/sklearn.py", line 1245, in <lambda>
    create_dmatrix=lambda **kwargs: DMatrix(nthread=self.n_jobs, **kwargs),
  ...
  File "/usr/local/lib/python3.9/site-packages/xgboost/data.py", line 772, in dispatch_data_backend
    return _from_pandas_df(data, enable_categorical, missing, threads,
  File "/usr/local/lib/python3.9/site-packages/xgboost/data.py", line 312, in _from_pandas_df
    data, feature_names, feature_types = _transform_pandas_df(
  File "/usr/local/lib/python3.9/site-packages/xgboost/data.py", line 256, in _transform_pandas_df
    _invalid_dataframe_dtype(data)
  File "/usr/local/lib/python3.9/site-packages/xgboost/data.py", line 236, in _invalid_dataframe_dtype
    raise ValueError(msg)
ValueError: DataFrame.dtypes for data must be int, float, bool or category.  When
categorical type is supplied, DMatrix parameter `enable_categorical` must
be set to `True`. Invalid columns:Deductions, Education, Employment, Gender, Marital, Occupation
```

This pipeline refactoring from "OHE style" to "categorical style" is provably correct, because these two configurations produce identical decision tree ensembles when fitted on the same platform using the same `tree_method` tree param.

The above statement is trivial to verify by diffing PMML files:

```
$ diff XGBoostAudit-OHE.pmml XGBoostAudit-categorical.pmml
```

The `diff` output shows changes to two lines of text.
First, the [`Header`](https://dmg.org/pmml/v4-4-1/Header.html#xsdElement_Header) element contains a different document creation timestamp.
Second, the data type of the "Deductions" field has changed from `boolean` to `string`.

The technical cause for the unexpected data type change remains unexplained.
Perhaps there is a slight incompatibility along the `scikit-learn`, `sklearn_pandas` and `pandas` library chain, which sorts itself out after a couple of package version updates.

In contrast, the above statement (about two decision tree ensembles being identical) is impossible to verify by diffing booster JSON files.

The booster JSON file references feature map entries by index.
However, this pipeline refactoring caused the embedded feature map to collapse in size (from 50 entries to 9 entries), thereby introducing large systematic shifts into index values.
For example, the "Education" field was mapped to a feature map entry range `[5, 20]` (ie. a list of 16 binary indicator features), but is now mapped to a sole feature map entry `[4]` (ie. a single categorical feature).

The `diff` output shows that the booster JSON file has been rewritten.
While technically correct, this observation is completely unhelpful when trying to understand the bigger picture.

## XGBoost version 1.6 (v1.6.2) ##

Deprecated the use of the `XGBClassifier.use_label_encoder` attribute, and added the [`XGBModel.max_cat_to_onehot`](https://github.com/dmlc/xgboost/blob/v1.6.2/python-package/xgboost/sklearn.py#L521) attribute.

Evidently, XGBoost is quickly getting better at categorical splits.
The OHE-based partitioning is considered outdated. The new default is [set-based aka optimal partitioning](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html#optimal-partitioning).

The `max_cat_tp_onehot` attribute allows the data scientist to choose one partitioning algorithm over the other.

The [default value of the `max_cat_to_onehot` tree param is 4](https://github.com/dmlc/xgboost/blob/v1.6.2/src/tree/param.h#L112-L115).
Typical values range from 2 to 20. Setting the value to an arbitrarily large number (eg. over 10'000) will effectively disable the set-based aka optimal partitioning.

Granted, one-hot encoding does not work particularly well with medium- and high-cardinality categorical features, especially if there are second-order effects involved (eg. sets of closely related category levels).

One-hot encoding faces additional headwinds with decision tree models, because it identifies and isolates significant category levels one by one, leading to long and thin branch growth.
For example, a member decision tree whose maximum depth is limited to 6 (the default value for the `max_depth` tree param) will be fully developed after coming in contact with a categorical feature that has six or more significant category levels.

Exporting the booster object is now also possible in [Universal Binary JSON](https://ubjson.org/) (UBJSON) data format:

``` python
# Note the ".ubj" filename extension!
classifier.save_model("Booster.ubj")
```

UBJSON is a binary data serialization format for JSON-compatible data structures.
It aims to improve data reading and writing speeds, and reduce data size. These capabilities are highly relevant, as the size of booster JSON files can reach tens to hundreds of megabytes.

## XGBoost version 1.7 (v1.7.3) ##

Added the [`XGBModel.max_cat_threshold`](https://github.com/dmlc/xgboost/blob/v1.7.3/python-package/xgboost/sklearn.py#L577) attribute.

This attribute limits the maximum set size for the set-based aka optimal partitioning.

The [default value of the `max_cat_threshold` tree param is 64](https://github.com/dmlc/xgboost/blob/v1.7.3/src/tree/param.h#L118-L123).

The XGBoost algorithm will consider a direct categorical split only if the number of category levels for a categorical feature has dropped below this value.
For example, the [USPS ZIP Code has over 40'000 codes](https://facts.usps.com/42000-zip-codes/).
When used as a categorical feature, then it would be useless to attempt a categorical split at a federal level (eg. sending 10'000 codes to the left and 30'000 codes to the right) or even at a state level. However, it would be very useful to attempt the same at county or city levels (eg. sending 10 codes to the left and 30 to the right).

The structure of member decision trees reflects how the original heterogeneous dataset is being divided into smaller and more homogenous subsets.
Continuous and low-cardinality categorical features (see the `max_cat_tp_onehot` tree param!) can appear at every branching level.
In contrast, medium- and high-cardinality categorical features initially stay on the sidelines. They appear gradually at deeper branching levels, where the details matter.

## PMML ##

The [JPMML-XGBoost](https://github.com/jpmml/jpmml-xgboost) library converts booster objects to the standardized Predictive Model Markup Language (PMML) representation.
This library supports all XGBoost versions and data formats (eg. Binary, JSON, UBJSON), and can be used either as a standalone command-line application or as an ML framework plugin.

Getting started with PMML is easy.
Simply replace the default Scikit-Learn's pipeline class with the `sklearn2pmml.pipeline.PMMLPipeline` class, and then pass the fitted pipeline object to the `sklearn2pmml.sklearn2pmml` utility function:

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", classifier)
])
pipeline.fit(X, y)

# Disable PMML optimizations
pipeline.configure(compact = False)

sklearn2pmml(pipeline, "XGBoostAudit.pmml")
````

The quality of PMML documents depends on the quality and completeness of the XGBoost model schema information.
For example, XGBoost versions 1.3 through 1.7 do not store category level values in the embedded feature map. It is impossible to generate meaningful categorical splits in such a situation, unless the embedded feature map is overriden with a more sophisticated external feature map.

The integration is much deeper within the JPMML ecosystem, as various JPMML platform libraries (eg. JPMML-SkLearn, JPMML-R, JPMML-SparkML) present the model schema as a live `org.jpmml.converter.Schema` object.

The JPMML-XGBoost library can expose one XGBoost model in different ways.
A PMML document can be optimized for explainability by expanding all decision tree leaves and maintaining complex feature values until the end. For example, generating [predicate elements](https://dmg.org/pmml/v4-4-1/TreeModel.html#xsdGroup_PREDICATE) that operate with temporal feature values (eg. `<SimplePredicate field="lastSeenDate" operator="lessOrEqual" value="2022-12-31"/>`).
Alternatively, a PMML document can be optimized for resource efficiency by compacting and simplifying it.

The conversion process can be guided using conversion options.

## Resources ##

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python scripts: [`train-OHE.py`]({{ "/resources/2023-02-06/train-OHE.py" | absolute_url }}) (XGBoost version 1.0 and newer) and [`train-categorical.py`]({{ "/resources/2023-02-06/train-categorical.py" | absolute_url }}) (XGBoost version 1.6 and newer)
