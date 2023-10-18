---
layout: post
title: "Training Scikit-Learn StatsModels pipelines"
author: vruusmann
keywords: scikit-learn sklearn2pmml patsy sklego statsmodels
related_posts:
  - 2023-10-15-sklearn_statsmodels_gridsearchcv_pipeline
---

[StatsModels](https://www.statsmodels.org) is a Python library that specializes in statistical and time-series analyses.

StatsModels and Scikit-Learn communities have co-existed since early 2010s.
Yet, there is virtually no co-operation between the two, even though the potential for synergy is high.

In essence, StatsModels provides low-level tools for formulating and testing statistical hypotheses, whereas Scikit-Learn provides a high-level framework for organizing those tools into coherent and transparent workflows.

[StatsModels' linear models](https://www.statsmodels.org/stable/user-guide.html#regression-and-linear-models) and [Scikit-Learn's linear models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) are compatible in their fit and predict behaviour, but address different application scenarios:

| &nbsp; | StatsModels | Scikit-Learn |
|--------|-------------|--------------|
| Audience | expert | non-expert |
| Methodology | analytical, statistics | brute-force ML |
| Objective | configurability | scalability and speed |
| Inspectability | "white box" | "black box" |
| Feedback | yes | no |

The biggest differentiator is model introspection capabilities.

StatsModels' linear models are metadata rich and lend themselves to scrutinization.
In contrast, Scikit-Learn's linear models can only be judged after their live predictive performance.
For example, all [model selection and evaluation](https://scikit-learn.org/stable/model_selection.html) takes place under the assumption that the quality of model configurations is directly proportional to the quality of its predictions.
There is no consideration of other aspects such as conformity or complexity.

## StatsModels workflow ##

A sample Scikit-Learn linear regression workflow:

``` python
from sklearn.linear_model import LinearRegression

lr_params = {...}

estimator = LinearRegression(**lr_params)

estimator.fit(X = X, y = y)

# Basic details
print(estimator.coef_)
print(estimator.intercept_)

yt = estimator.predict(X = X)
```

The same, after porting from Scikit-Learn to StatsModels:

``` python
from statsmodels.api import OLS

lr_params = {...}

model = OLS(endog = y, exog = X)

results = model.fit(**lr_params)

# Basic details
print(results.params)
# Advanced details
print(results.summary())

results.remove_data()

yt = results.predict(exog = X)
```

A modeling task can be split into **task specification** and **task outcome** parts.

Scikit-Learn keeps these two parts together in a singular estimator object.
Calling the `fit(X, y)` method with the training dataset updates the object from the initial state to the fitted state, and enables the use of `predict(X)` and `score(X, y_expected)` methods.
The fitted state is very lean, and does not include any trace of the training dataset.

In contrast, StatsModels keeps these two parts separate between `statsmodels.base.model.Model` and `statsmodels.base.model.Results` objects, respectively.

The `Model` object is instantiated directly. It holds the base configuration and the training dataset.
According to StatsModels conventions, the label data (aka dependent variable aka response) is passed as the `endog` argument, and the feature data (aka independent variables aka observations) is passed as the `exog` argument.

A model class can declare one or more fit methods:
* `fit(method, ...)` - Full fit.
* `fit_regularized(method, ...)` - Regularized fit.
* `fit_constrained(constraints, ...)` - Constrained fit.

Each fit method is backed by one or more modeling algorithms.
When calling a fit method, then it will be necessary to pass a rather long and detailed supplementary configuration there (on top of the base configuration) in order to achieve the best performance.

The fit produces a `Results` object, which holds a (back-)reference to the parent `Model` object, and a bunch of estimated parameters.

The `Model` object loses its relevance from this point forward.
All the analysis, testing and prediction functionality is commandable using the `Results` object's methods.

## Scikit-Learn StatsModels wrappers ##

The working principle of a StatsModels wrapper class:

``` python
from sklearn.base import BaseEstimator

class StatsModelsEstimator(BaseEstimator):

  def __init__(self, model_class, **init_params):
    self.model_class = model_class
    self.init_params = init_params

  def fit(self, X, y, **fit_params):
    self.model_ = self.model_class(endog = y, exog = X, **self.init_params)
    fit_method = fit_params.pop("fit_method", "fit")
    self.results_ = getattr(self.model_, fit_method)(**fit_params)
    return self

  def predict(self, X, **predict_params):
    return self.results_.predict(exog = X, **predict_params)
```

The `StatsModelsEstimator` constructor requests to know the model's class and its base configuration.
This information is stored as-is.

The `model_class` argument must be some `Model` class reference.
It does not matter if this reference is based on the fully qualified name (eg. `statsmodels.regression.linear_model.OLS`) or some alias (eg. `statsmodels.api.OLS`).

The `init_params` argument is a list of keyword arguments to be passed to the `Model` constructor at a later time.
This list must not include the `endog` or `exog` arguments, or any formula interface-related arguments.

The `fit(X, y)` method performs all the heavy-lifting.
First, it constructs the `Model` object by complementing the base configuration with the newly available training dataset.
After that, it determines the fit method by popping a special-purpose `fit_method` fit param, and then calls it using the remainder of fit params.

The fitted state consists of `model_` and `results_` attributes.

Sample workflow:

``` python
from statsmodels.api import OLS

estimator = StatsModelsEstimator(OLS)
estimator.fit(X, y, fit_method = "fit_regularized")

print(estimator.results_.summary())
```

It takes around ten lines of Python code to achieve basic interoperability.
However, it will take many times that to reduce the entry barrier from the expert level to the beginner-to-intermediate level, and achieve advanced interoperability so that third-party libraries can accept StatsModels' linear models as drop-in replacements for Scikit-Learn's linear models.

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides `sklearn2pmml.statsmodels.StatsModelsClassifier` and `sklearn2pmml.statsmodels.StatsModelsRegressor` models that strive towards maximal Scikit-Learn API compliance.

Selected highlights:

* The `StatsModelsEstimator.fit_intercept` attribute.
* `StatsModelsEstimator.coef_` and `StatsModelsEstimator.intercept_` attributes.
* The `StatsModelsEstimator.remove_data()` method. Reduces the size of the model by evicting the training dataset.
* The `StatsModelsClassifier.classes_` attribute.
* `StatsModelsClassifier.predict(X)` and `StatsModelsClassifier.predict_proba(X)` methods.

Both `StatsModelsClassifier` and `StatsModelsRegressor` classes are registered with the JPMML-SkLearn library.
However, the conversion to the PMML representation is only possible if the fitted state operates with `Model` and `Results` subclasses that are supported by the underlying JPMML-StatsModels library.

## Scikit-Learn pipeline templates ##

### Model formula interface

StatsModels is heavily influenced by the R language.
One clear telltale is the habit of specifying modeling tasks in compact symbolic form.

StatsModels relies on the [`patsy`](https://github.com/pydata/patsy) package for its R-compatible re-implementation of [model formulae](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html) and [model design matrices](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/model.matrix.html).

The parsing and interpretation of model formulae is tailored for linear models' needs.
For example, the model formula string is assumed to include an implicit intercept term, and the `C()` categorical encoding operator drops the first category level.

A sample Patsy linear regression workflow:

``` python
from sklearn2pmml.statsmodels import StatsModelsRegressor
from statsmodels.api import OLS

import pandas
import patsy

df = pandas.read_csv(...)

# Model formula sides
lhs = ...
rhs = ...

y, X = patsy.dmatrices(lhs + " ~ " + rhs, df, return_type = "dataframe")

regressor = StatsModelsRegressor(OLS, fit_intercept = False)
regressor.fit(X, y)

print(regressor.results_.summary())
```

The same, in Scikit-Learn pipeline form:

``` python
from sklearn.pipeline import Pipeline
from sklearn2pmml.statsmodels import StatsModelsRegressor
from sklego.preprocessing import PatsyTransformer
from statsmodels.api import OLS

import pandas

df = pandas.read_csv(...)

rhs = ...

pipeline = Pipeline([
  ("patsy_transformer", PatsyTransformer(rhs, return_type = "dataframe")),
  ("regressor", StatsModelsRegressor(OLS, fit_intercept = False))
])
pipeline.fit(df, df[...])

regressor = pipeline._final_estimator

print(regressor.results_.summary())
```

The [`sklego`](https://github.com/koaning/scikit-lego) package provides the `sklego.preprocessing.PatsyTransformer` transformer, which learns the model design matrix configuration (`patsy.DesignInfo` class) during fitting, and then applies it during subsequent transforms.
The transformation behaviour is perfectly repeatable across time and space, including in situations where the testing dataset is a subset of the training dataset (eg. a single data sample).

The `PatsyTransformer` transformer only deals with the right-hand side of the formula.
Its output is explicitly converted from `patsy.DesignMatrix` to `pandas.DataFrame` in order to ensure smooth passage between Patsy and Scikit-Learn realms.

The first column of the pre-processed dataset is labelled "Intercept" and is filled with constant `1.0` values.
The `StatsModelsRegressor` model is expressly prohibited from introducing an extra constant column by settings its `fit_intercept` attribute to `False`.

### Data matrix interface

The data matrix interface is about retrofitting existing Scikit-Learn pipelines.

Model substitution is straightforward.
The only pitfall relates to regularized fit methods, where StatsModels' linear models expect direct commanding via fit params:

``` python
#from sklearn.linear_model import ElasticNet
from sklearn2pmml.statsmodels import StatsModelsRegressor
from statsmodels.api import OLS

pipeline = Pipeline([
  ("mapper", ...),
  #("regressor", ElasticNet()),
  ("regressor", StatsModelsRegressor(OLS, fit_intercept = True))
])

fit_params = {
  "regressor__fit_method" : "fit_regularized",
  "regressor__method" : "elastic_net"
}

pipeline.fit(X, y, **fit_params)
```

StatsModels summaries are readily available.
However, the default print-out is quite meaningless, because it displays a `const` term followed by a long list of `x1`, `x2`, etc. terms.

The situation can be fixed by proper feature identification.

Scikit-Learn transformers offer rudimentary feature naming support using the `get_feature_names_out()` method.
Calling this method on the last transformer step of a fitted pipeline yields a list of strings that should align perfectly against the above list of anonymized terms.

This process can be fully automated in Scikit-Learn version 1.2.0 and newer using the [set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html):

``` python
from sklearn2pmml.statsmodels import StatsModelsRegressor
from statsmodels.api import OLS

pipeline = Pipeline([
  ("mapper", ...),
  ("regressor", StatsModelsRegressor(OLS, fit_intercept = True))
])
pipeline.set_output(transform = "pandas")

fit_params = {
  "regressor__fit_method" : "fit_regularized",
  "regressor__method" : "elastic_net"
}

pipeline.fit(X, y, **fit_params)

regressor = pipeline._final_estimator

print(regressor.results_.summary())
```

The `Pipeline.set_output()` method propagates the same configuration to all steps.
It should be called right after construction.

Scikit-Learn generates less intuitive feature names than Patsy.

The results of some transformer step can be highlighted by prepending a unique feature name prefix to them.
This can be implemented by wrapping the transformer into a `sklearn.compose.ColumnTransformer` object:

``` python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

def make_prefixer(transformer, name):
  return ColumnTransformer([
    (name, transformer, lambda X: X.columns.values)
  ])

interactor = make_prefixer(PolynomialFeatures(degree = 2, interaction_only = True), "ix1")
```

The interaction features of the `interactor` object will now be easily identifiable in StatsModels summaries by their `"ix1__"` prefix.
