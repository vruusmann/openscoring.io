---
layout: post
title: "Converting Scikit-Learn hyperparameter-tuned pipelines to PMML"
author: vruusmann
keywords: scikit-learn sklearn2pmml
---

The behaviour of Scikit-Learn estimators is controlled using hyperparameters.
Feature transformers and selectors perform deterministic computations that take a very limited number of very transparent hyperparameters.
In contrast, models perform non-deterministic computations (numerical optimization) that take a much larger number of rather obscure hyperparameters.
Some of them control the complexity of the learned model object, whereas some others control the quality and speed of the learning process itself.

Scikit-Learn estimators assign reasonable default values to hyperparameters in their constructors.
This facilitates prototyping work, where the goal is to establish the structure of a pipeline by quickly adding or modifying steps.
However, the default configuration is hardly ever the optimal one.

There is no analytic procedure for determining the best configuration from scratch, or even comparing two configurations goodness-wise.
In practice, the most common way of finding a good configuration is to generate many configurations, and rank them on the basis of their predictive performance on a testing dataset.

The [Model Selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) module provides meta-estimators and utility functions for developing robust solutions in this area.

In brief, a data scientist defines the template pipeline and the associated hyperparameter space.
The latter is a mapping between parameter names and parameter value ranges (a list of preselected values, or a distribution function).
If the dimensionality of the hyperparameter space is low, and the gradation of all individual dimensions is directly enumerable, then it is possible to perform exhaustive search using the `GridSearchCV` meta-estimator.
In all other cases, it is possible to perform random sampling using the `RandomizedSearchCV` meta-estimator.

## Single estimator (aka local) tuning ##

If the pipeline contains just one tuneable estimator, then the tuning work should be performed locally, by wrapping this estimator in its current place into a search meta-estimator:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

mapper = DataFrameMapper(
  [(cat_column, [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
  [([cont_column], [ContinuousDomain(), StandardScaler()]) for cont_column in cont_columns]
)

classifier = LogisticRegression(multi_class = "ovr", penalty = "elasticnet", solver = "saga", max_iter = 1000)

param_grid = {
  "l1_ratio" : [0.7, 0.8, 0.9]
}

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", GridSearchCV(estimator = classifier, param_grid = param_grid))
])
pipeline.fit(df_X, df_y)
pipeline.verify(df_X.sample(n = 5))

sklearn2pmml(pipeline, "GridSearchAudit.pmml")
```

Both `GridSearchCV` and `RandomizedSearchCV` meta-estimators split the original dataset into training and validation subsets.
As a result, the fit method of the tuneable estimator is exposed to less data records than the fit methods of all the other estimators in the pipeline.
For example, in the above Python code, the `LogisticRegression.fit(X, y)` method is called with roughly 80% of data records (the training subset of the original dataset), whereas `LabelBinarizer.fit(X)` and `StandardScaler.fit(X)` methods are called with 100% of data records (full original dataset).
Data scientists may want to compensate for this effect, especially when working with smaller and more heterogeneous datasets.

## Pipeline (aka global) tuning ##

If the pipeline contains multiple tuneable estimators, then the tuning work should be performed globally, by wrapping the complete pipeline into a search meta-estimator:

``` python
from sklearn.feature_selection import SelectKBest

mapper = DataFrameMapper(
  [(cat_column, [CategoricalDomain(invalid_value_treatment = "as_is"), LabelBinarizer()]) for cat_column in cat_columns] +
  [([cont_column], [ContinuousDomain(invalid_value_treatment = "as_is"), StandardScaler()]) for cont_column in cont_columns]
)

selector = SelectKBest()

classifier = LogisticRegression(multi_class = "ovr", penalty = "elasticnet", solver = "saga", max_iter = 1000)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("selector", selector),
  ("classifier", classifier)
])

param_grid = {
  "selector__k" : [10, 20, 30],
  "classifier__l1_ratio" : [0.7, 0.8, 0.9]
}

searcher = GridSearchCV(estimator = pipeline, param_grid = param_grid)
searcher.fit(df_X, df_y)

best_pipeline = searcher.best_estimator_
best_pipeline.verify(df_X.sample(n = 5))

sklearn2pmml(best_pipeline, "GridSearchAudit.pmml")
```

The `GridSearchCV` meta-estimator can be regarded as a workflow execution engine.
It takes a template pipeline, performs the search, and returns a hyperparameter-tuned clone of this template pipeline as a `best_estimator_` attribute.

All hyperparameter spaces are collected into a single map.
They are kept separate from one another logically by prefixing parameter names with the step identifier.

The search meta-estimator is still splitting the original dataset into two subsets.
However, the split happens before the workflow execution enters the `(PMML)Pipeline.fit(X, y)` method, so all estimators in the pipeline are exposed to the same number of data records.

If the span of a validation subset exceeds that of a training subset, then the corresponding cross-validation fold shall fail with a value error:

```
Traceback (most recent call last):
  File "train-global.py", line 41, in <module>
    searcher.fit(df_X, df_y)
  File "/usr/local/lib/python3.7/site-packages/sklearn/model_selection/_search.py", line 712, in fit
    self._run_search(evaluate_candidates)
  File "/usr/local/lib/python3.7/site-packages/sklearn/model_selection/_search.py", line 1153, in _run_search
    evaluate_candidates(ParameterGrid(self.param_grid))
  File "/usr/local/lib/python3.7/site-packages/sklearn/model_selection/_search.py", line 691, in evaluate_candidates
    cv.split(X, y, groups)))
  ...
  File "/usr/local/lib/python3.7/site-packages/sklearn/pipeline.py", line 613, in score
    Xt = transform.transform(Xt)
  File "/usr/local/lib/python3.7/site-packages/sklearn_pandas/dataframe_mapper.py", line 377, in transform
    return self._transform(X)
  File "/usr/local/lib/python3.7/site-packages/sklearn_pandas/dataframe_mapper.py", line 306, in _transform
    Xt = transformers.transform(Xt)
  File "/usr/local/lib/python3.7/site-packages/sklearn/pipeline.py", line 555, in _transform
    Xt = transform.transform(Xt)
  File "/usr/local/lib/python3.7/site-packages/sklearn2pmml/decoration/__init__.py", line 119, in transform
    self._transform_invalid_values(X, invalid_value_mask)
  File "/usr/local/lib/python3.7/site-packages/sklearn2pmml/decoration/__init__.py", line 103, in _transform_invalid_values
    raise ValueError("Data contains {0} invalid values".format(numpy.count_nonzero(where)))
ValueError: Employment: Data contains 1 invalid values
```

It is possible to suppress this sanity check by changing the value of the `Domain.invalid_value_treatment` attribute from `return_invalid` to `as_is`.

## Resources ##

* Dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python scripts: [`train-local.py`]({{ "/resources/2019-12-25/train-local.py" | absolute_url }}) and [`train-global.py`]({{ "/resources/2019-12-25/train-global.py" | absolute_url }})
