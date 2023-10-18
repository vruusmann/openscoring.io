---
layout: post
title: "Training Scikit-Learn GridSearchCV StatsModels pipelines"
author: vruusmann
keywords: scikit-learn sklearn2pmml statsmodels tuning
related_posts:
  - 2023-03-28-sklearn_statsmodels_pipeline
  - 2023-05-03-converting_sklearn_subclass_pmml
  - 2019-12-25-converting_sklearn_gridsearchcv_pipeline_pmml
---

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides Scikit-Learn style wrappers for StatsModels classification, ordinal classification and regression models:

* `sklearn2pmml.statsmodels.StatsModelsClassifier`
* `sklearn2pmml.statsmodels.StatsModelsOrdinalClassifier`
* `sklearn2pmml.statsmodels.StatsModelsRegressor`

Wrapper classes are maximally generic by design.
For example, the `StatsModelsRegressor` wrapper can accommodate any StatsModels regression model, any StatsModels version.

This genericity is achieved using [Python Arbitrary Keyword Arguments](https://docs.python.org/3/glossary.html#term-argument) (aka `**kwargs`) mechanisms.
Wrapper class methods accept any keyword arguments. They get packed into a singular `dict`-type helper param, which is then dispatched to the right StatsModels model method at the right time.

Such "param aggregation" approach works fine with most common Scikit-Learn workflows.
However, it falls short with advanced workflows, where individual Scikit-Learn estimators need to be queried and/or updated on a param-by-param basis.

The most prominent example is hyperparameter tuning.

## Training ##

A hyperparameter tuning workflow can be summarized as follows:
1. The end user provides a template estimator object and declares all its tunable params.
2. The tuner makes a clone of the template estimator object and updates the initial values of one or more tunable params with new values.
3. The tuner scores the updated estimator object. The updated estimator object is kept if it out-scores the previous best estimator object, and is discarded otherwise.
4. The tuner repeats stages #2 and #3 until the stop criterion is met.

In the second workflow stage, candidate Scikit-Learn estimators are fabricated using the [`sklearn.base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) utility function.
The fabrication algorithm constructs a new unfitted estimator object irrespective if the "template" was a fitted or unfitted estimator object (ie. "selective clone" rather than "full clone" semantics).

The tunable param set is determined via the `get_params` method.
The default implementation returns the constructor params of the estimator class.

Wrapper classes declare two named params `model_class` and `fit_intercept`.
They are typically set by a human expert depending on the nature of the modeling task.
However, there are no technical restrictions to varying them using grid search means in order to conduct a quick AutoML-like StatsModels model selection experiment:

``` python
from sklearn.model_selection import GridSearchCV
from sklearn2pmml.statsmodels import StatsModelsRegressor
from statsmodels.api import GLM, OLS, WLS

pipeline = make_statsmodels_pipeline(StatsModelsRegressor(OLS))

ctor_params_grid = {
  "regressor__model_class" : [GLM, OLS, WLS],
  "regressor__fit_intercept" : [True, False]
}

tuner = GridSearchCV(pipeline, param_grid = ctor_params_grid, verbose = 3)
tuner.fit(X, y)

print(tuner.best_estimator_)
```

Any attempt to call the `GridSearchCV.fit(X, y)` method with an expanded param grid shall fail with a value error.

### Task-specific subclassing

The canonical approach to enabling tunable params is by declaring them one-by-one as constructor params, and assigning them to Python class attributes (with exactly the same name) in the constructor body.

For example, defining a wrapper subclass for tuning the `alpha` and `L1_wt` params of the [`OLS.fit_regularized`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit_regularized.html) method:

``` python
class TunableStatsModelsRegressor(StatsModelsRegressor):

  def __init__(self, model_class, fit_intercept = True, alpha = 0.01, L1_wt = 1, **init_params):
    super(TunableStatsModelsRegressor, self).__init__(model_class = model_class, fit_intercept = fit_intercept, **init_params)
    self.alpha = alpha
    self.L1_wt = L1_wt

  def fit(self, X, y, **fit_params):
    super(TunableStatsModelsRegressor, self).fit(X, y, alpha = self.alpha, L1_wt = self.L1_wt, **fit_params)
    return self
```

### Task-agnostic subclassing

Code reusability can be improved by aggregating all tunable params into a singular `dict`-type helper param.

However, this approach qualifies as a hack, because it takes advantage of the notion that the tuner is willing to perform update operations also on virtual params.
Such loophole exists in Scikit-Learn versions 1.0.X through 1.2.X. It may be closed in newer versions.

``` python
class TunableStatsModelsRegressor(StatsModelsRegressor):

  def __init__(self, model_class, fit_intercept = True, tune_params = {}, **init_params):
    super(TunableStatsModelsRegressor, self).__init__(model_class = model_class, fit_intercept = fit_intercept, **init_params)
    self.tune_params = tune_params

  def set_params(self, **params):
    super_params = dict([(k, params.pop(k)) for k, v in dict(**params).items() if k in ["model_class", "fit_intercept", "tune_params"]])
    super(TunableStatsModelsRegressor, self).set_params(**super_params)
    setattr(self, "tune_params", dict(**params))

  def fit(self, X, y, **fit_params):
    super(TunableStatsModelsRegressor, self).fit(X, y, **self.tune_params, **fit_params)
    return self
```

The above class self-reports `model_class`, `fit_intercept` and `tune_params` params, but is open to perform an update operation on any param.

This "discrepancy" is powered by a custom `set_params` method.
Update operations that target self-reported params are dispatched to the superclass' `set_params` method.
All other update operations are treated as Python item assignment operations against the `tune_params` param.

The combined StatsModels model selection and model hyperparameter tuning experiment succeeds with either flavour of `TunableStatsModelsRegressor`:

``` python
pipeline = make_statsmodels_pipeline(TunableStatsModelsRegressor(OLS))

ctor_params_grid = {
	"regressor__fit_intercept" : [True, False]
}

regfit_params_grid = {
	"regressor__alpha" : loguniform(1e-2, 1).rvs(5),
	"regressor__L1_wt" : uniform(0, 1).rvs(5)
}

tuner = GridSearchCV(pipeline, param_grid = {**ctor_params_grid, **regfit_params_grid}, verbose = 3)
tuner.fit(X, y, regressor__fit_method = "fit_regularized")

print(tuner.best_estimator_)
```

## Deployment ##

Hyperparameter tuning is 100% training-time phenomenon, which should not cause any complications at later model lifecycle phases.
Any kind of subclassing violates this principle, because it creates a need to package and distribute model class definition(s) alongside the model object.

When the model deployment happens in the Python environment, then the solution is to transform the `TunableStatsModelsRegressor` object into a new object for which there is a stable and reusable class definition available.

The `TunableStatsModelsRegressor` class does not hold any fitted state beyond that of its `StatsModelsRegressor` parent class.
Therefore, as another hack, the model object can be made easily recognizable to its future users by simply re-assigning its `__class__` attribute:

``` python
import joblib

best_pipeline = tuner.best_estimator_

best_regressor = best_pipeline._final_estimator
best_regressor.__class__ = StatsModelsRegressor

joblib.dump(best_pipeline, "GridSearchAuto.pkl")
```

The proper way of model class conversion is to re-fit a new pipeline object using the best param set.
If the latter is a mix of constructor and fit method params, then it must be partitioned into two disjoint subsets:

``` python
import joblib

best_params = dict(tuner.best_params_)

best_pipeline = make_statsmodels_pipeline(StatsModelsRegressor(OLS, fit_intercept = best_params.pop("regressor__fit_intercept")))
best_pipeline.fit(X, y, **best_params, regressor__fit_method = "fit_regularized")

joblib.dump(best_pipeline, "GridSearchAuto.pkl")
```

Model deployment in non-Python environments might seem impossible due to extensive StatsModels, Scikit-Learn and Python API dependencies.

No worries, because the [Java PMML API](https://github.com/jpmml) software project provides a full stack of Java tools and libraries for untangling and converting arbitrary complexity Python ML artifacts to the Predictive Model Markup Language (PMML) representation.
Dealing with the `best_pipeline` object in a fully automated fashion does not pose any substantial challenge.

There is one minor configuration issue related to the fact that the [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) library recognizes and supports wrapper classes, but not their ad hoc subclasses.

It is possible to avoid tedious model class conversion operation by declaring model class equivalence (ie. "treat `TunableStatsModelsRegressor` objects the same as `StatsModelsRegressor` objects") using a custom class mapping:

``` python
from sklearn2pmml import load_class_mapping, make_class_mapping_jar, sklearn2pmml
from sklearn2pmml.util import fqn

default_mapping = load_class_mapping()

# Map the ad hoc subclass to the same JPMML-SkLearn converter class as the parent class
statsmodels_mapping = {
	fqn(TunableStatsModelsRegressor) : default_mapping[fqn(StatsModelsRegressor)]
}

extension_jar = "TunableStatsModelsRegressor.jar"

make_class_mapping_jar(statsmodels_mapping, extension_jar)

sklearn2pmml(tuner.best_estimator_, "GridSearchAuto.pmml", user_classpath = [extension_jar])
```

## Resources ##

* Dataset: [`auto.csv`]({{ "/resources/data/auto.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2023-10-15/train.py" | absolute_url }})