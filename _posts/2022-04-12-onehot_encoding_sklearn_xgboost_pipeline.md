---
layout: post
title: "One-hot encoding categorical features in Scikit-Learn XGBoost pipelines"
author: vruusmann
keywords: scikit-learn xgboost sklearn2pmml data-categorical data-missing
---

Creating a Scikit-Learn pipeline that feeds categorical data into XGBoost estimator is surprisingly tricky, because there are rather significant mismatches in their conceptual and technical designs.

On one hand, Scikit-Learn aims to be the simplest and most generic ML framework that can integrate reasonably well with any Python AI/ML library.
This simplicity has been achieved via major trade-offs.
For example, Scikit-Learn is currently not very comfortable dealing with rich dataset schemas.
The majority of its algorithms operate on dense numeric arrays.
There are efforts being made towards retrofitting existing transformers to pass through missing and non-numeric values as-is, but the overall experience is still far from coherent and consistent.

On the other hand, XGBoost aims to be the most powerful and flexible gradient boosting algorithm.
No one and no thing is perfect, though.
Historically, the biggest pain point for XGBoost has been its sub-par support for categorical features.
The team is well aware of this, and has committed to solving it step by step.
All the necessary groundwork, and the first categorical encoding algorithm based on a popular one-hot encoding (OHE) scheme has been recently completed as documented in [dmlc/xgboost-6503](https://github.com/dmlc/xgboost/issues/6503).
More sophisticated encoding schemes are expected to follow.

Nevertheless, a Scikit-Learn based XGBoost pipeline is composable for any version combination of the two, for any dataset.

This blog post details the technical background, and provides quick and resource-efficient recipes for the following combinations:

* External OHE for XGBoost versions 0.60 -- 1.3, for dense datasets.
* External OHE for XGBoost versions 0.60 -- 1.3, for sparse datasets.
* Model-internal OHE for XGBoost version 1.4 and newer

### Categorical data in XGBoost

The results of optimization work depend on the correct identification of limiting steps ("bottlenecks"), and finding ways to overcome or bypass them.

Software engineering is about finding solutions in the software-controlled part of the stack.
If the solution appears to reside in the hardware or meatware parts of the stack, then it should be delegated to more appropriate engineering teams.

The standard XGBoost-via-Python software stack stands as follows:

1. [XGBoost API](https://xgboost.readthedocs.io/en/stable/dev/files.html). The core library, plus device-specific (CPU, GPU) acceleration libraries. Defines C++ language data structures and algorithms.
2. [Python Learning API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training). Python language wrappers for the XGBoost API. Data conversion and transfer from Python environment to C++ environment.
3. [Scikit-Learn API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn). Scikit-Learn compatible Python language wrappers for the Python Learning API. Embedding XGBoost estimators into Scikit-Learn workflows.

XGBoost C++ algorithms operate with a specialized `xgboost::DMatrix` data structure only.

In Python environment, the training data exists in Python-specifc data structures such as Pandas' data frames or Numpy arrays.
When the Python Learning API layer passes data to the XGBoost API layer, then Python data structures get left behind into the Python environment, and a new and detached `xgboost::DMatrix` object is created in the C++ environment.

Typical data flow:

``` python
from xgboost import DMatrix

import xgboost

df = pandas.read_csv(...)

# X is pandas.DataFrame
X = df[[feature_col1, feature_col2, .., feature_coln]]
y = df[label_col]

transformer = make_transformer()

# Xt is numpy.ndarray or pandas.DataFrame
Xt = transformer.fit_transform(X)

# dmat is xgboost.DMatrix
dmat = DMatrix(data = Xt, label = y)

# In this point in the program execution flow,
# there are three "data stores" in memory: X, Xt and dmat
booster = xgboost.train(dtrain = dmat, ...)
```

The total memory requirement of a Scikit-Learn pipeline can be estimated by summing the sizes of `X`, `Xt` and `dmat` datasets

The "deep copying" from `Xt` dataset to `dmat` dataset effectively doubles the memory requirements of XGBoost pipelines in comparison with standard Scikit-Learn pipelines.

The situation is manageable when the training dataset consists of continuous features.
However, the situation can easily get out of hand when there are categorical features present, especially when they are high-cardinality ones.

The crux of the matter is encoding categorical features into (pseudo-)numeric features using Scikit-Learn transformers in order to make them acceptable for the final XGBoost estimator.

The most common and robust approach is [one-hot encoding](https://en.wikipedia.org/wiki/One-hot), which transforms a single categorical feature into a list of binary indicator features.
In the above example, when the `X` dataset contains categorical features, then the number of columns in `Xt` and `dmat` datasets would increase considerably (one binary indicator column per category level).

Starting from XGBoost version 1.3 it is possible to replace external OHE with model-internal OHE.
This functionality can be activated by changing the data type of relevant feature columns to [`pandas.CategoricalDtype`](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html) (aka `category`).
The Scikit-Learn API layer should pass through `category` dtype columns unchanged.
They will be automatically detected in the Python Learning API layer, and their intrinsic integer-encoded values will be transferred to the XGBoost API layer.
In the above example, all `X`, `Xt` and `dmat` datasets would contain the same number of categorical feature columns.

By eliminating a high-cost but low-value data transformation in the Python environment, it will be possible to run existing experiments much faster, or start running new and much bigger experiments.

### Legacy workflow: external OHE

The legacy workflow implements OHE in the Scikit-Learn API layer, and therefore supports all XGBoost versions, including all the pre-1.0 ones.

However, dense and sparse datasets require different external OHE approaches.

The canonical representation of a one-hot encoded categorical value is an n-element bit vector, where n is the cardinality of the categorical feature.
For a non-missing value, this bit vector contains a single `1`-valued element (the "on" bit) and n-1 `0`-valued elements ("off" bits). For a missing value, this bit vector contains n `NaN`-valued elements.

Transformation results for a sparse color list `["red", "green", None, "blue"]`:

| input value | "blue" bit | "green" bit | "red" bit |
|---|---|---|---|
| "red" | `0.0` | `0.0` | `1.0` |
| "green" | `0.0` | `1.0` | `0.0` |
| `None` | `NaN` | `NaN` | `NaN` |
| "blue" | `1.0` | `0.0` | `0.0` |

**Any other representation is prone to mis-interpretation by XGBoost C++ algorithms**.

For example, many Scikit-Learn transformers produce bit vectors where the "off" bits are denoted by `NaN`-valued elements (rather than the canonical `0`-valued elements):

| input value | "blue" bit | "green" bit | "red" bit |
|---|---|---|---|
| "red" | `NaN` | `NaN` | `1.0` |
| "green" | `NaN` | `1.0` | `NaN` |
| `None` | `NaN` | `NaN` | `NaN` |
| "blue" | `1.0` | `NaN` | `NaN` |

This representation is acceptable for Scikit-Learn estimators, which only track the location of the `1`-valued "on" bit.
However, this is unacceptable for LightGBM, XGBoost or any other missing-value aware estimator, which additionally tracks the locations of `0`-valued "off" bits and `NaN`-valued "info not available" bits.

The correctness of a custom encoding can be verified by comparing its predictions against the ones of the most simplistic, known-good encoding.
The best reference is canonical OHE results stored in a dense array.

For example, the predictions of an XGBoost estimator must come out identical (ie. numerically equivalent) when first applied to a custom-encoded sparse dataset, and then to its "densified" copy:

``` python
import numpy

transformer = make_custom_sparse_transformer()
estimator = make_xgboost_estimator()

Xt = transformer.fit_transform(X)

estimator.fit(Xt, y)

yt_sparse = estimator.predict(Xt)
yt_dense = estimator.predict(Xt.todense())

# Must not raise an AssertionError
numpy.testing.assert_array_equal(yt_sparse, yt_dense)
```

##### Dense datasets

Standard Scikit-Learn transformers such as `sklearn.preprocessing.LabelBinarizer` and `sklearn.preprocessing.OneHotEncoder` are suitable for working with dense datasets only, because their output is inherently two-state (ie. binary).

The `OneHotEncoder` might be preferable due to its multi-column support and high configurability.
First and foremost, `OneHotEncoder` must be configured to produce dense arrays (ie. `sparse = False`).
Optionally, `OneHotEncoder` could be configured to change the array data type to the most economical 1-byte integer data type (ie. `dtype = numpy.uint8`) in order to keep the size of the `Xt` dataset as small as possible.

Suggested transformer:

``` python
from sklearn.preprocessing import OneHotEncoder

def make_dense_legacy_transformer(cat_cols, cont_cols):
  return ColumnTransformer([
    ("cont", "passthrough", cont_cols),
    ("cat", OneHotEncoder(sparse = False, dtype = numpy.uint8), cat_cols)
  ], sparse_threshold = 1.0)
```

Memory usage with `make_dense_legacy_transformer` function:

| dtype | `X` bytesize | `Xt` bytesize | `dmat` bytesize | total bytesize |
|---|---|---|---|---|
| `numpy.uint8` | 121536 | **89253** | 737314 | 948'103 |
| `numpy.float32` | 121536 | 357012 | 737314 | 1'215'862 |
| `numpy.float64` | 121536 | **714144** | 737314 | 1'572'994 |

The memory usage of `xgboost::DMatrix` objects cannot be measured directly in Python environment, because its payload is held opaquely in the C++ environment.
Therefore, the measurement is performed in a roundabout way, by dumping the `xgboost::DMatrix` object into a binary file in local filesystem, and then querying the file size.

The results show that by changing the `OneHotEncoder.dtype` attribute from the default `numpy.float64` dtype to `numpy.uint8` dtype it is possible to achieve an eight times reduction of `Xt` dataset size.

The size of the `dmat` dataset stays constant.
An `xgboost::DMatrix` object that is constructed based on a dense Python array appears to transfer data into a dense C++ array, where the cells are dimensioned for 64-bit floating point values.

Data scientists who work with large datasets often try to "compress" `Xt` and `dmat` datasets by various means.
The fact that the XGBoost API layer accepts anything that comes from higher layers without warnings or errors does not offer any validation that the "compression" was indeed implemented correctly.

**One common mistake** is using `OneHotEncoder` in its default `sparse = True` configuration.
The resulting sparse matrix contains only `1`- and `NaN`-valued elements (and not a single `0`-valued element), which leads XGBoost C++ algorithms to think that it is dealing with massive amounts of missing values.

**Another common mistake** is producing a dense `Xt` dataset, and then converting it to a sparse `dmat` dataset by specifying `missing = 0` either in the `xgboost.DMatrix` constructor or any of `XGBModel.fit(X, y)` methods:

``` python
transformer = make_dense_legacy_transformer()

Xt = transformer.fit_transform(X)

# Incorrect compression!
dmat = DMatrix(data = Xt, label = y, missing = 0)
```

##### Sparse datasets

There are no standard Scikit-Learn transformers suitable for working with sparse datasets as-is.
The prevailing workaround is filling in missing values via imputation.
However, while appropriate with Scikit-Learn estimators, it should be generally avoided with XGBoost estimators, which can handle missing values natively, and in fact can learn and reveal interesting relationships around them.

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides a `sklearn2pmml.preprocessing.PMMLLabelBinarizer` transformer that can produce both two-state dense arrays (`sparse_output = False`) or tri-state sparse matrices (`sparse_output = True`).

Suggested transformer:

``` python
from sklearn2pmml.preprocessing import PMMLLabelBinarizer

def make_sparse_legacy_transformer(cat_cols, cont_cols):
  return ColumnTransformer(
    [(cont_col, "passthrough", [cont_col]) for cont_col in cont_cols] +
    [(cat_col, PMMLLabelBinarizer(sparse_output = True), [cat_col]) for cat_col in cat_cols]
  , sparse_threshold = 1.0)
```

Memory usage with `make_sparse_legacy_transformer` function:

| dtype | `X` bytesize | `Xt` bytesize | `dmat` bytesize | total bytesize |
|---|---|---|---|---|
| `numpy.float64` | 121536 | **138748** | 103146 | 363'430 |

The sparsity of the `X` dataset is around 28.2 percent (4'286 cells out of 8 * 1'899 = 15'192 cells).
The sparsity of the `Xt` dataset is around 87.8 percent (78'347 cells out of 47 * 1'899 = 89'253 cells), which gives it excellent compressibility properties.

An `xgboost::DMatrix` object that is constructed based on a sparse Python matrix appears to transfer data into a sparse C++ matrix as well.
Interestingly enough, XGBoost can find a data layout that reduces memory usage significantly.

### Modern workflow: model-internal OHE

XGBoost version 1.3 and newer support model-internal OHE.

If a dataset contains categorical features that are suitable for OHE then, from now on, this transformation should be moved out of high-level Python API layers (eg. `OneHotEncoder.transform(X)` or `pandas.get_dummies(X)`) into the low-level XGBoost API layer.
Doing things closer to the core opens up new and vastly superior capabilities such as GPU-accelerated histogram generation.

Typical data flow:

``` python
from xgboost import DMatrix

df = pandas.read_csv(...)

X = df[[feature_col1, feature_col2, .., feature_coln]]
y = df[label_col]

# Cast categorical features into "category" data type
for cat_col in cat_cols:
  X[cat_col] = X[cat_col].astype("category")

dmat = DMatrix(data = Xt, label = y, enable_categorical = True)

booster_params = {
  "tree_method" : "gpu_hist"
}

# In this point in the program execution flow,
# there are two "data stores" in memory: X, and dmat
booster = xgboost.train(params = booster_params, dtrain = dmat, ...)
```

The cast to `pandas.CategoricalDtype` data type (aka `category`) is missing value-aware.
Therefore, the same Scikit-Learn pipeline can be used both with dense and sparse datasets.

Memory usage:

| Experiment | `X` bytesize | `dmat` bytesize | total bytesize |
|---|---|---|---|
| dense | 121536 | **129807** | 251'343 |
| sparse | 121536 | **103323** | 224'859 |

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* "Audit-NA" dataset: [`audit-NA.csv`]({{ "/resources/data/audit-NA.csv" | absolute_url }})
* Python scripts: [`measure.py`]({{ "/resources/2022-04-12/measure.py" | absolute_url }}), [`train.py`]({{ "/resources/2022-04-12/train.py" | absolute_url }}) and [`util.py`]({{ "/resources/2022-04-12/util.py" | absolute_url }})