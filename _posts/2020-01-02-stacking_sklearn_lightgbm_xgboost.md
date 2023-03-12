---
layout: post
title: "Stacking Scikit-Learn, LightGBM and XGBoost models"
author: vruusmann
keywords: scikit-learn lightgbm xgboost sklearn2pmml data-categorical data-missing
---

Latest Scikit-Learn releases have made significant advances in the area of ensemble methods.

[Scikit-Learn version 0.21](https://scikit-learn.org/stable/whats_new/v0.21.html) introduced `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` classes, which implement histogram-based decision tree ensembles.
They are based on a completely new `TreePredictor` decision tree representation.
The claimed benefits over the traditional `Tree` decision tree representation include support for missing values and the ability to process bigger datasets faster.

[Scikit-Learn version 0.22](https://scikit-learn.org/stable/whats_new/v0.22.html) introduced `StackingClassifier` and `StackingRegressor` classes, which aggregate multiple child estimators into an integral whole using a parent (aka final) estimator.
Stacking is closely related to voting.
The main difference is about how the weights for individual child estimators are obtained.
Namely, stacking estimators are "active" as they learn optimal weights autonomously during training, whereas voting estimators are "passive" as they expect optimal weights to be supplied.

Scikit-Learn implements two stacking modes.
In the default non-passthrough mode, the parent estimator is limited to seeing only the predictions of child estimators (`predict_proba` for classifiers and `predict` for regressors).
In the passthrough mode, the parent estimator also sees the input dataset.

## Stacking homogeneous estimators ##

The pipeline is very simple and straightforward when dealing with homogeneous estimators.

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import numpy
import pandas

df = pandas.read_csv("auto.csv")

cat_columns = ["cylinders", "model_year", "origin"]
cont_columns = ["acceleration", "displacement", "horsepower", "weight"]

df_X = df[cat_columns + cont_columns]
df_y = df["mpg"]

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), OneHotEncoder()]) for cat_column in cat_columns] +
  [([cont_column], [ContinuousDomain(), StandardScaler()]) for cont_column in cont_columns]
)

estimator = StackingRegressor([
  ("dt", DecisionTreeRegressor(max_depth = 7, random_state = 13)),
  ("mlp", MLPRegressor(hidden_layer_sizes = (7), solver = "lbfgs", max_iter = 2000, random_state = 13))
], final_estimator = LinearRegression(), passthrough = True)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("estimator", estimator)
])
```

The qualifier "homogeneous" means that all child estimators have the same data pre-processing requirements.
The opposite of "homogeneous" is "heterogeneous", which means that different child estimators have different data pre-processing requirements.

Consider, for example, the preparation of continuous features.
Linear models assume that the magnitude of continuous values is roughly the same.
Decision tree models do not make such an assumption, because they can identify an optimal split threshold value for a continuous feature irrespective of its transformation status (original scale vs. transformed scale).
Owing to this discrepany, linear models and decision tree models (and ensembles thereof) are incompatible with each other by default.

It is often possible to simplify a "heterogeneous" collection of estimators to a "homogeneous" one by performing data pre-processing following the strictest requirements.
Linear models and decision tree models become compatible with each other after all continuous features have been scaled (ie. a requirement of linear models, which does not make any difference for decision tree models).

## Stacking heterogeneous estimators ##

The pipeline needs considerable redesign when dealing with heterogeneous estimators.

Stacking LightGBM and XGBoost estimators is challenging due to their different categorical data pre-processing requirements.

LightGBM performs the histogram-based binning of categorical values internally, and therefore expects categorical features to be kept as-is, or at most be encoded into categorical integer features using the `LabelEncoder` transformer.
XGBoost does not have such capabilities, and therefore expects categorical features to be binarized using either `LabelBinarizer` or `OneHotEncoder` transformers.

The "homogenisation" of LightGBM and XGBoost estimators is possible by enforcing the binarization of categorical features.
However, this reduces the predictive performance of LightGBM.
For more information, please refer to the blog post about [converting Scikit-Learn LightGBM pipelines to PMML]({% post_url 2019-04-07-converting_sklearn_lightgbm_pipeline_pmml %}).

The solution is to perform data pre-processing for each child estimator (and in the passthrough mode, also for the parent estimator) separately:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn2pmml.pipeline import PMMLPipeline

mapper = DataFrameMapper(..)

estimator = StackingClassifier([
  ("first", Pipeline(..)),
  ("second", Pipeline(..)),
  ("third", Pipeline(..))
], final_estimator = Pipeline(..))

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("estimator", estimator)
])
```

If all child pipelines perform common data pre-processing work, then it should be extracted into the first step of the pipeline.
In this exercise, it is limited to capturing domain of features using `CategoricalDomain` and `ContinuousDomain` decorators.

The initial column transformer changes the representation of the dataset from `pandas.DataFrame` to 2-D Numpy array, which is lacking adequate column-level metadata (eg. names, data types) for setting up subsequent column transformers.
A suitable array descriptor is created manually, by copying the value of the `DataFrame.dtypes` attribute, and changing its index from column names to column positions:

``` python
from sklearn.compose import ColumnTransformer

df = pandas.read_csv("audit-NA.csv", na_values = ["N/A", "NA"])

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

df_X = df[cat_columns + cont_columns]
df_y = df["Adjusted"]

dtypes = df_X.dtypes

mapper = ColumnTransformer(
  [(cat_column, CategoricalDomain(), [cat_column]) for cat_column in cat_columns] +
  [(cont_column, ContinuousDomain(), [cont_column]) for cont_column in cont_columns]
)

dtypes = Series(dtypes.values, index = [0, 1, 2, 3, 4, 5, 6])
```

Column transformers for LightGBM and XGBoost child pipelines can be constructed using `sklearn2pmml.preprocessing.lightgbm.make_lightgbm_column_transformer` and `sklearn2pmml.preprocessing.xgboost.make_xgboost_column_transformer` utility functions, respectively.

LightGBM estimators are able to detect categorical features based on their data type.
However, when dealing with more complex datasets, then it is advisable to overrule this functionality by supplying the indices of categorical features manually.
This is typically done by specifying a `categorical_feature` (prefixed with one or more levels of step identifiers) parameter to the `(PMML)Pipeline.fit(X, y, **fit_params)` method.
Unfortunately, this route is currently blocked, because the fit methods of `StackingClassifier` and `StackingRegressor` classes do not support the propagation of fit parameters.
The workaround is to pass the `categorical_feature` parameter directly to the constructor.

Constructing the LightGBM child pipeline:

``` python
from lightgbm import LGBMClassifier
from sklearn2pmml.preprocessing.lightgbm import make_lightgbm_column_transformer

lightgbm_mapper, lightgbm_categorical_feature = make_lightgbm_column_transformer(dtypes, missing_value_aware = True)
lightgbm_pipeline = Pipeline([
  ("mapper", lightgbm_mapper),
  ("classifier", LGBMClassifier(n_estimators = 31, max_depth = 3, random_state = 13, categorical_feature = lightgbm_categorical_feature))
])
```

Constructing the XGBoost child pipeline:

``` python
from sklearn2pmml.preprocessing.xgboost import make_xgboost_column_transformer
from xgboost import XGBClassifier

xgboost_mapper = make_xgboost_column_transformer(dtypes, missing_value_aware = True)
xgboost_pipeline = Pipeline([
  ("mapper", xgboost_mapper),
  ("classifier", XGBClassifier(n_estimators = 31, max_depth = 3, random_state = 13))
])
```

The Scikit-Learn child pipeline has exactly the same data pre-processing requirements as the XGBoost one (ie. continuous features should be kept as-is, whereas categorical features should be binarized).
Currently, the corresponding column transformer needs to be set up manually.
In future Scikit-Learn releases, when the fit methods of `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` classes add support for sparse datasets, then it should be possible to reuse the `make_xgboost_column_transformer` utility function here.

``` python
from sklearn.experimental import enable_hist_gradient_boosting # noqa

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn2pmml.preprocessing import PMMLLabelBinarizer

sklearn_mapper = ColumnTransformer(
  [(str(cat_index), PMMLLabelBinarizer(sparse_output = False), [cat_index]) for cat_index in range(0, len(cat_columns))] +
  [(str(cont_index), "passthrough", [cont_index]) for cont_index in range(len(cat_columns), len(cat_columns + cont_columns))]
, remainder = "drop")
sklearn_pipeline = Pipeline([
  ("mapper", sklearn_mapper),
  ("classifier", HistGradientBoostingClassifier(max_iter = 31, max_depth = 3, random_state = 13))
])
```

Stacking provides an interesting opportunity to rank LightGBM, XGBoost and Scikit-Learn estimators based on their predictive performance.
The idea is to grow all child decision tree ensemble models under similar structural constraints, and use a linear model as the parent estimator (`LogisticRegression` for classifiers and `LinearRegression` for regressors).
The importance of child estimators is then proportional to their estimated coefficient values.

To further boost signal over noise, the stacking is performed in non-passthrough mode and its cross-validation functionality is disabled by supplying a no-op cross-validation generator:

``` python
from sklearn.linear_model import LogisticRegression

# See https://stackoverflow.com/a/55326439
class DisabledCV:

  def __init__(self):
    self.n_splits = 1

  def split(self, X, y, groups = None):
    yield (numpy.arange(len(X)), numpy.arange(len(y)))

  def get_n_splits(self, X, y, groups=None):
    return self.n_splits

final_estimator = LogisticRegression(multi_class = "ovr", random_state = 13)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("ensemble", StackingClassifier([
    ("lightgbm", lightgbm_pipeline),
    ("xgboost", xgboost_pipeline),
    ("sklearn", sklearn_pipeline)
  ], final_estimator = final_estimator, cv = DisabledCV(), passthrough = False))
])
```

A fitted `PMMLPipeline` object can be converted to a PMML XML file using the `sklearn2pmml.sklearn2pmml` utility function.
However, it is highly advisable to first enhance it with verification data (for automated quality-assurance purposes) by calling the `PMMLPipeline.verify(X)` method with a representative sample of the input dataset:

``` python
from sklearn2pmml import sklearn2pmml

pipeline = PMMLPipeline(..)
pipeline.fit(df_X, df_y)

pipeline.verify(df_X.sample(n = 10))

sklearn2pmml(pipeline, "StackingEnsemble.pmml")
```

## Resources ##

* "Audit-NA" dataset: [`audit-NA.csv`]({{ "/resources/data/audit-NA.csv" | absolute_url }})
* "Auto" dataset: [`auto.csv`]({{ "/resources/data/auto.csv" | absolute_url }})
* Python scripts: [`train-classification.py`]({{ "/resources/2020-01-02/train-classification.py" | absolute_url }}) and [`train-regression.py`]({{ "/resources/2020-01-02/train-regression.py" | absolute_url }})
