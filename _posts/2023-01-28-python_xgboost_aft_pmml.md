---
layout: post
title: "Training Python-based XGBoost accelerated failure time models"
author: vruusmann
keywords: scikit-learn xgboost sklearn2pmml data-categorical data-missing
---

Survival analysis is a subtype of regression analysis, which models time durations (eg. time to an event of interest).

XGBoost supports both Cox proportional hazards (Cox PH) and accelerated failure time (AFT) algorithms.
The training is currently possible via the low-level [Python Learning API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training), but not via the high-level [Scikit-Learn API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn).
This limitation can be tracked as [XGBoost-7292](https://github.com/dmlc/xgboost/issues/7292).

The solution is to train XGBoost survival models using the Python Learning API, and then migrate them to the Scikit-Learn API for the eventual productionization.

## Data canonicalization

Loading the ["lung" dataset](https://lifelines.readthedocs.io/en/latest/lifelines.datasets.html#lifelines.datasets.load_lung):

``` python
from lifelines.datasets import load_lung

df = load_lung()

print(df.dtypes)
print(df.isna().sum())
```

The loaded data matrix contains ten numeric columns:

| Name | Role | Loaded `dtype` | Refined `dtype` | Missing |
|------|------|----------------|-----------------|---------|
| `inst` | feature | `float64` | `pandas.Int64Dtype` | 1 |
| `time` | proto-label | `int64` | `int64` | 0 |
| `status` | proto-label | `int64` | `int64` | 0 |
| `age` | feature | `int64` | `int64` | 0 |
| `sex` | feature | `int64` | `int64` |  0 |
| `ph.ecog` | feature | `float64` | `pandas.Int64Dtype` | 1 |
| `ph.karno` | feature | `float64` | `pandas.Int64Dtype` | 1 |
| `pat.karno` | feature | `float64` | `pandas.Int64Dtype` | 3 |
| `meal.cal` | feature | `float64` | `pandas.Int64Dtype` | 47 |
| `wt.loss` | feature | `float64` | `pandas.Int64Dtype` | 14 |

The `time` and `status` columns encode time durations, leaving the other eight columns as features.

All feature data are inherently integer-like. However, they are stored in `float64` data type columns in order to accommodate missing value placeholders in the form of `float("NaN")`.

Refining column data types:

``` python
from pandas import Int64Dtype

cols = df.columns.tolist()

for col in cols:
  has_missing = pandas.isnull(df[col]).any()
  df[col] = df[col].astype(Int64Dtype() if has_missing else int)
```

## Data pre-processing

XGBoost versions 1.5 and newer can work with the canonicalized "lung" dataset as-is.
There is no longer any technical reason for imputing missing values or encoding categorical values.

In fact, this is a welcome development, as Scikit-Learn is rather inept at transforming sparse categorical features into legacy XGBoost-compatible representation, as discussed in detail in an earlier blog post about [one-hot encoding categorical features in Scikit-Learn XGBoost pipelines]({% post_url 2022-04-12-onehot_encoding_sklearn_xgboost_pipeline %}).

The only data pre-processing action that is needed is casting the data type of `inst` and `sex` columns from integer to [`pandas.CategoricalDtype`](https://pandas.pydata.org/docs/reference/api/pandas.CategoricalDtype.html):

``` python
df["inst"] = df["inst"].astype("category")
df["sex"] = df["sex"].astype("category")
```

However, the above code is not suitable for productionization!

A quick cast using the `category` data type alias creates a new `CategoricalDtype` object on each and every call.
Its "business state" is a mapping from category levels to category indices.
If a testing dataset does not have exactly the same set of unique category levels as the training dataset, then the mapping will be different, and along with it all the predictions.

Constructing a casting transformer that is robust towards dataset variations:

``` python
from pandas import CategoricalDtype
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

cat_cols = ["inst", "sex"]
cont_cols = ["age", "ph.ecog", "ph.karno", "pat.karno", "meal.cal", "wt.loss"]

def make_cat_dtype(x):
  categories = pandas.unique(x)
  # Drop null-like category levels
  categories = numpy.delete(categories, pandas.isnull(categories))
  return CategoricalDtype(categories = categories, ordered = False)

inst_dtype = make_cat_dtype(df["inst"])
sex_dtype = make_cat_dtype(df["sex"])

transformer = DataFrameMapper(
  [(["inst"], CategoricalDomain(dtype = inst_dtype))] +
  [(["sex"], CategoricalDomain(dtype = sex_dtype))] +
  [([cont_col], ContinuousDomain(dtype = numpy.float32)) for cont_col in cont_cols]
, input_df = True, df_out = True)
```

The cast to `CategoricalDtype` data type is performed using the `sklearn2pmml.decoration.CategoricalDomain` decorator. Alternatively, it could be performed using the `sklearn2pmml.preprocessing.CastTransformer` transformer.
Using decorators is preferable, because they capture valuable metadata about the training dataset (valid value space, univariate statistics, etc.) which serves as model documentation.

This cast is possible with Pandas' data containers (eg. `pandas.DataFrame` and `pandas.Series`), but not with Numpy arrays and matrices.

The propagation of Pandas' data containers through Scikit-Learn pipeline has been challenging.
Scikit-Learn version 1.2.0 introduces the [set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html), which allows to configure the output data container type for all transformer subclasses using the [`TransformerMixin.set_output(transform)`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin.set_output) method.

It will take some time until the set_output API gets propagated through the ecosystem.
Until then, the tried and tested `sklearn_pandas.DataFrameMapper` meta-transformer class is the way to go.

Unlike categorical features, continuous features do not require any casting or transformation here.
Nevertheless, they are also filtered through the `sklearn2pmml.decoration.ContinuousDomain` decorator in order to capture metadata.

In the above example, all columns are declared one by one, even if the associated transformation supports multi-column input.
This is needed for preserving the original column names.

Re-coding the label as pairs of scalar values:

``` python
def make_aft_label(time, status):
  time_lower = time

  time_upper = time.copy()
  time_upper[status == 0] = float("+Inf")

  return (time_lower, time_upper)
```

According to XGBoost conventions, uncensored labels are encoded as `(<value>, <value>)`, whereas right-censored labels are encoded as `(<value>, +Inf)`.

## Training via Python Learning API

Constructing an [`xgboost.DMatrix`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix) object:

``` python
from xgboost import DMatrix

Xt = transformer.fit_transform(df)

time_lower, time_upper = make_aft_label(df["time"], df["status"])

dmat = DMatrix(
  # Features
  data = Xt,
  # Label
  label_lower_bound = time_lower, label_upper_bound = time_upper,
  missing = float("NaN"), 
  enable_categorical = True
)
```

Training an [`xgboost.Booster`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster) object, and saving it in JSON data format:

``` python
import xgboost

params = {
  "objective" : "survival:aft",
  "eval_metric" : "aft-nloglik",
  "max_depth" : 3,
  "tree_method" : "hist"
}

booster = xgboost.train(params = params, dtrain = dmat, num_boost_round = 31)
booster.save_model("booster.json")
```

**Important**: The computation of categorical splits must be activated by setting the [`tree_method` tree param](https://xgboost.readthedocs.io/en/stable/treemethod.html) to some approximated training algorithm such as `approx` or `hist` (`gpu_hist` on GPU implementations).
If this tree param is set to `exact` (default for small datasets such as the "lung" dataset), then the computation yields continuous splits, irrespective of the underlying data type.
XGBoost does not issue any warnings about the fallback.

When in doubt, it is possible to open the booster JSON file in a text editor, and skim over `categories_nodes`, `categories_segments` and `categories_sizes` properties.
With categorical features around, one would expect to see a healthy proportion of non-empty list values.
It would be a clear cause for concern if they were all empty.

## Making predictions via Scikit-Learn API

Scikit-Learn pipeline is a composite of transformers and estimators.
It provides atomic pickling, and unified `fit_transform(X)` and `fit_predict(X, y)` API methods for dealing with arbitrary complexity workflows.

Constructing a pipeline from available components:

``` python
from sklearn.pipeline import Pipeline
from xgboost import Booster
from xgboost.sklearn import XGBRegressor

regressor = XGBRegressor()
regressor._Booster = Booster(model_file = "booster.json")

pipeline = Pipeline([
  ("transformer", transformer),
  ("regressor", regressor)
])
```

As noted earlier, survival analysis is a subtype of regression analysis.
This makes it possible to wrap an AFT booster object into an `xgboost.XGBRegressor` object, and have it directly respond to `XGBRegressor.predict(X)` method calls.

## Pipeline verification

Typically, a `Pipeline` object is constructed using unfitted components, and fitted right thereafter. This is convenient and makes a very strong guarantee that all pipeline steps work together harmoniously.
However, there are still plenty of situations where one or more pipeline steps defy the unified API, or make use of extra information or functionality that is not available within the `Pipeline.fit(X, y)` method scope.

When a `Pipeline` object is constructed from pre-fitted components, then there are no guarantees in place. For example, the `Pipeline` constructor does not even check if the output dimensions of one step match the input dimensions of the next step.

The correctness of unknown origin and unknown status `Pipeline` objects can only be assessed empirically, by calling their predict methods (eg. `predict(X)`, `predict_proba(X)`) with adequate verification datasets, and comparing actual results against expected results.

Asserting equivalence between Python Learning API and Scikit-Learn API predictions:

``` python
def check_predict(expected, actual, rtol, atol):
  isclose = numpy.isclose(expected, actual, rtol = rtol, atol = atol, equal_nan = False)
  num_conflicts = numpy.sum(isclose == False)
  if num_conflicts:
    for idx, status in enumerate(isclose):
      if not status:
        print("{} != {}".format(expected[idx], actual[idx]))
    raise ValueError("Found {} conflicting prediction(s)".format(num_conflicts))
  print("All correct")

# Reference values
booster_time = booster.predict(dmat)

# To-be-checked values
pipeline_time = pipeline.predict(df)

check_predict(booster_time, pipeline_time, 1e-6, 1e-3)
```

The AFT booster object makes predictions using the same `DMatrix` object that it was trained on, whereas the pipeline object makes predictions starting from the canonicalized dataset.
If there were any discrepancies between their predictions, then it would be natural to put more trust in the former (the "expected" side), and less in the latter (the "actual" side).

Arrays of floating-point numeric values can be compared element-wise for equivalence using the [`numpy.isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) utility function.

The algorithm is controlled by two parameters.
First, relative tolerance (the `rtol` argument) is a workflow property (the precision of the default numeric data type, plus the order of "mathematical complexity" of transformations).
Second, absolute tolerance (the `atol` argument) is a dataset property.

The XGBoost algorithm is based on `float32` data type, which limits the relative tolerance to about `1E-6 .. 1E-7` range.

The label of the "lung" dataset is survival time in days, with a typical range from 100 to 2000.
The absolute tolerance value of `1E-3` is estimated by multiplying the selected relative tolerance `1E-6` with the selected characteristic label value of `1000`.

In other words, two survival times are considered to be equivalent, if they agree with each other in the order of "full minutes or better".

## PMML

Wrapping the pipeline object into a `PMMLPipeline` object:

``` python
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

pmml_pipeline = make_pmml_pipeline(pipeline, active_fields = (cat_cols + cont_cols), target_fields = ["time"])

Xt_imp = booster.get_score(importance_type = "weight")
print(Xt_imp)

# Transform dict to list
Xt_imp = [Xt_imp[col] for col in pmml_pipeline.active_fields]

regressor.pmml_feature_importances_ = numpy.asarray(Xt_imp)

df_verif = df[pmml_pipeline.active_fields].sample(10)

pmml_pipeline.verify(df_verif, precision = 1e-6, zeroThreshold = 1e-3)

sklearn2pmml(pmml_pipeline, "XGBoostAFTLung.pmml")
```

It is advisable to use the `sklearn2pmml.make_pmml_pipeline(obj)` utility function for constructing a `PMMLPipeline` object from pre-fitted components, because it is programmed to perform exactly the same extra object initialization work as the `PMMLPipeline.fit(X, y)` method is doing.
For example, setting the `PMMLPipeline.active_fields` and `PMMLPipeline.target_fields` attributes.

The `PMMLPipeline` object is enhanced with verification data in order to auto-discover any regressions that might arise from migration from Python platform to other language platforms.
PMML implements [model verification](https://dmg.org/pmml/v4-4-1/ModelVerification.html) similarly to the `numpy.isclose` utility function.
The main difference is terminological, as relative tolerance and absolute tolerance are called "precision" and "zero threshold", respectively.

## Resources

* Python script: [`train.py`]({{ "/resources/2023-01-28/train.py" | absolute_url }})