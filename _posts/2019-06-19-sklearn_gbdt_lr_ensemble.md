---
layout: post
title: "Extending Scikit-Learn with GBDT+LR ensemble models"
author: vruusmann
keywords: scikit-learn lightgbm xgboost sklearn2pmml data-categorical
---

Logistic regression (LR) is often the go-to choice for binary classification.
Owing to extreme simplicity, LR models are fast to train and easy to deploy, and readily lend themselves for human interpretation.

The predictive performance of LR models depends on the quality and sophistication of feature engineering.
There are two major work areas.
First, delineating and generating the intended feature space. LR algorithms operate on the feature space they are given.
They are not designed to independently discover non-linearities along individual dimensions, or interactions between multiple dimensions.
Second, filtering down the feature space.
Specialized LR algorithms can prioritize and eliminate dimensions using regularization. However, the most common ones estimate coefficients for all dimensions of the feature space.

Facebook Research has demonstrated how feature engineering can be automated using a gradient boosted decision tree (GBDT) model: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)

The idea is to train a GBDT model on a raw feature space and collect and examine the "decision paths" of its member decision tree models.
A decision path which operates on a single feature can be regarded as a non-linear transformation on it (eg. binning a continuous feature to a pseudo-categorical feature). A decision path which operates on multiple features can be regarded as an interaction between them.

GBDT algorithms typically grow shallow decision trees.
Shallow trees contain short decision paths, which generally lead to easily interpretable derived features.

A boosting algorithm can be swapped for a bagging algorithm.
Random forest (RF) algorithms typically grow much deeper decision trees.
Longer decision paths lead to more complex derived features (eg. interactions between multiple non-linearly transformed features), which lose in interpretability but gain in information content.
For example, discovering cliffs and other anomalies in the decision space by observing which derived features become associated with extreme node scores.

### Scikit-Learn perspective

Scikit-Learn documentation dedicates a separate page to GBDT plus LR (GBDT+LR) ensemble models: [Feature transformations with ensembles of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)

While the concept and its implementation are discussed in great detail, there is no reusable GBDT+LR estimator available within Scikit-Learn.
Interested parties are either expected to copy-paste the example code, or rely on third-party libraries.

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package version 0.47.0 introduced `sklearn2pmml.ensemble.GBDTLRClassifier` and  `sklearn2pmml.ensemble.GBDTLMRegressor` ensemble models to address this deficiency.

### PMML perspective

The Predictive Model Markup Language (PMML) provides standardized data structures for representing all common data pre- and post-processing operations and model types, including the GBDT model type and the LR model type.

If all the parts of a GBDT+LR model are PMML compatible, then it should follow that the GBDT+LR model itself is PMML compatible too?
The answer is a definite yes. Better yet, the PMML representation of a GBDT+LR model is reducible to an ordinary GBDT model, which leads to significant conversion- and run-time savings.

The reduction is based on the realization that GBDT+LR is a mechanism for replacing original GBDT leaf node scores with LR coefficients (and the GBDT base score with the LR intercept).
Scikit-Learn does not provide an API for modifying fitted decision trees.
The workaround is to make individual leaf nodes addressable using the one-hot-encoding approach (the `OneHotEncoder.categories` attribute is a list of arrays; the size of the list equals the number of decision trees in the GBDT; the size of each array equals the number of leaf nodes in the corresponding decision tree), and then assigning a new score to each address (the `LogisticRegression.coef_` attribute is an array whose size equals the flat-mapped size of the `OneHotEncoder.categories` attribute).

The PMML representation does not need such layer of indirection, because it is possible to replace leaf node scores in place.

The [JPMML-Model](https://github.com/jpmml/jpmml-model) library provides Visitor API for traversing, updating and transforming PMML class model objects.
In the current case, the Visitor API is used to transform the GBDT side of the GBDT+LR model to a regression-type boosting model. All leaf nodes are assigned new score values as extracted from the LR side.

### Example workflow

The GBDT+LR workflow is much simpler than traditional workflows.
Specifically, there is no need to perform dedicated feature engineering work, because the GBDT+LR estimator will do it automatically and in a very thorough manner.

Boilerplate for assembling and fitting a GBDT+LR pipeline using user-specified `gbdt` and `lr` components:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.ensemble import GBDTLRClassifier
from sklearn2pmml.pipeline import PMMLPipeline

import pandas

df = pandas.read_csv(..)

# The names of categorical and continuous feature columns
cat_columns = [...]
cont_columns = [...]

# The name of the label column
label_column = ..

def make_fit_gbdtlr(gbdt, lr):
  mapper = DataFrameMapper(
    [([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
    [(cont_columns, ContinuousDomain())]
  )
  classifier = GBDTLRClassifier(gbdt, lr)
  pipeline = PMMLPipeline([
    ("mapper", mapper),
    ("classifier", classifier)
  ])
  pipeline.fit(df[cat_columns + cont_columns], df[label_column])
  return pipeline
```

The most common configuration is to use `GradientBoostingClassifier` as the `gbdt` component.
The "boosting" behaviour can be promoted by growing a larger number of shallower decision trees.

``` python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import sklearn2pmml

pipeline = make_fit_gbdtlr(GradientBoostingClassifier(n_estimators = 499, max_depth = 2), LogisticRegression())
sklearn2pmml(pipeline, "GBDT+LR.pmml")
```

Conversely, the "bagging" behaviour can be promoted by growing a smaller number of deeper decision trees.
The `GBDTLRClassifier` ensemble model accepts any PMML compatible classifier as the `gbdt` component.
For example, switching from `GradientBoostingClassifier` to alternative classifier classes such as `ExtraTreesClassifier` or `RandomForestClassifier` would reduce the risk of overfitting:

``` python
from sklearn.ensemble import RandomForestClassifier

pipeline = make_fit_gbdtlr(RandomForestClassifier(n_estimators = 31, max_depth = 6), LogisticRegression())
sklearn2pmml(pipeline, "RF+LR.pmml")
```

The [XGBoost](https://github.com/dmlc/xgboost) plugin library provides an `xgboost.XGBClassifier` model, which can be used as a drop-in replacement for Scikit-Learn classifier classes:

``` python
from xgboost import XGBClassifier

pipeline = make_fit_gbdtlr(XGBClassifier(n_estimators = 299, max_depth = 3), LogisticRegression())
sklearn2pmml(pipeline, "XGB+LR.pmml")
```

The [LightGBM](https://github.com/microsoft/LightGBM) plugin library provides a `lightgbm.LGBMClassifier` model.
One of its major selling points is proper support for categorical features.
If the training dataset contains a significant number of (high-cardinality-) categorical features, then the above `make_fit_gbdtlr` utility function should be tailored to maintain this information.

As discussed in [a recent blog post]({% post_url 2019-04-07-converting_sklearn_lightgbm_pipeline_pmml %}), the fit method of LightGBM estimators takes an optional `categorical_feature` fit parameter.
The next challenge is about passing this parameter to a `LGBMClassifier` object, which is contained in the `GBDTLRClassifier` object, which is in turn contained in the `(PMML)Pipeline` object.
The solution follows Scikit-Learn conventions.
Namely, the fit method of the `GBDTLRClassifier` class also takes fit parameters, which are passed on to the correct component based on the prefix.

Boilerplate for assembling and fitting an LightGBM+LR pipeline:

``` python
from sklearn.preprocessing import LabelEncoder

def make_fit_lgbmlr(gbdt, lr):
  mapper = DataFrameMapper(
    [([cat_column], [CategoricalDomain(), LabelEncoder()]) for cat_column in cat_columns] +
    [(cont_columns, ContinuousDomain())]
  )
  classifier = GBDTLRClassifier(gbdt, lr)
  pipeline = PMMLPipeline([
    ("mapper", mapper),
    ("classifier", classifier)
  ])
  # The 'gbdt' component can be addressed using the `classifier__gbdt` prefix
  # The 'lr' component can be addressed using the `classifier__lr` prefix
  pipeline.fit(df[cat_columns + cont_columns], df[label_column], classifier__gbdt__categorical_feature = range(0, len(cat_columns)))
  return pipeline
```

Sample usage:

``` python
from lightgbm import LGBMClassifier

pipeline = make_fit_lgbmlr(LGBMClassifier(n_estimators = 71, max_depth = 5), LogisticRegression())
sklearn2pmml(pipeline, "LGBM+LR.pmml")
```

Both XGBoost and LightGBM classifiers support missing values.
When working with sparse datasets, then it is possible to make `make_fit_gbdtlr` and `make_fit_lgbmlr` utility functions missing value-aware by replacing the default `LabelBinarizer` and `LabelEncoder` transformers with `sklearn2pmml.preprocessing.PMMLLabelBinarizer` and `sklearn2pmml.preprocessing.PMMLLabelEncoder` transformers, respectively.

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2019-06-19/train.py" | absolute_url }})