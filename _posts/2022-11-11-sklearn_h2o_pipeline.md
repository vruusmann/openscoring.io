---
layout: post
title: "Training Scikit-Learn H2O.ai pipelines"
author: vruusmann
keywords: h2o scikit-learn sklearn2pmml data-categorical data-missing
---

[H2O.ai](https://h2o.ai/) is an alternative ML framework, which is built with maximum in-memory scaling properties in mind.
Scikit-Learn users may find H2O.ai interesting when working with datasets that exceed the logical and physical limits of a desktop computer.

Sure, it is often possible to ignore the problem for extended periods of time by renting a bigger and faster computer.
But the fact remains that Scikit-Learn is not a "Big Data"-oriented ML framework by design, and some new tools and new ways of doing things are likely to yield much better results.

### Estimator upgrade from Scikit-Learn to H2O.ai

H2O.ai is written in Java, and typically runs in a managed server (whether on-premise or "cloud").
Python users can interact with an H2O.ai server using the [`h2o`](https://github.com/h2oai/h2o-3/tree/master/h2o-py) package.

Every H2O.ai algorithm is encapsulated into an `h2o.estmators.H2OEstimator` subclass:

| Scikit-Learn class | H2O.ai class |
|--------------------|--------------|
| `sklearn.cluster.KMeans` | `h2o.estimators.kmeans.H2OKMeansEstimator` |
| `sklearn.ensemble.IsolationForest` | `h2o.estimators.isolation_forest.H2OIsolationForestEstimator` |
| `sklearn.ensemble.GradientBoostingClassifier`, `GradientBoostingRegressor` | `h2o.estimators.gbm.H2OGradientBoostingEstimator` |
| `sklearn.ensemble.RandomForestClassifier`, `RandomForestRegressor` | `h2o.estimators.random_forest.H2ORandomForestEstimator` |
| `sklearn.isotonic.IsotonicRegression` | `h2o.estimators.isotonicregression.H2OIsotonicRegressionEstimator` |
| `sklearn.linear_model.LinearRegression`, `LogisticRegression` | `h2o.estimators.glm.H2OGeneralizedLinearEstimator` |
| `sklearn.naive_bayes.GaussianNB` | `h2o.estimators.naive_bayes.H2ONaiveBayesEstimator` |
| `sklearn.svm.SVC`, `SVR` | `h2o.estimators.psvm.H2OSupportVectorMachineEstimator` |

According to [H2O.ai modeling documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html), all these classes have Scikit-Learn style `fit(X, y)` and `predict(X)` methods readily available.

This suggests that it should be possible to insert H2O.ai estimator objects into standard Scikit-Learn pipelines:

``` python
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn.pipeline import Pipeline

import pandas

df = pandas.read_csv("audit.csv")

X = df[[column for column in df.columns if column != "Adjusted"]]
y = df["Adjusted"]

pipeline = Pipeline([
  ("classifier", H2OGeneralizedLinearEstimator())
])
pipeline.fit(X, y)
```

The `Pipeline` constructor succeeds, but the subsequent `Pipeline.fit(X, y)` method call raises a rather obscure attribute error:

```
Traceback (most recent call last):
  File "train.py", line 23, in <module>
    pipeline.fit(X, y)
  File "/usr/local/lib/python3.9/site-packages/sklearn/pipeline.py", line 382, in fit
    self._final_estimator.fit(Xt, y, **fit_params_last_step)
  File "/usr/local/lib/python3.9/site-packages/h2o/estimators/estimator_base.py", line 481, in fit
    training_frame = X.cbind(y) if y is not None else X
  File "/usr/local/lib/python3.9/site-packages/pandas/core/generic.py", line 5902, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'cbind'
```

### Manual data upload to H2O.ai server

Closer inspection of the [`H2OEstimator.fit(X, y)`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2o.estimators.estimator_base.H2OEstimator.fit) method signature reveals that it expects both `X` and `y` arguments to be of `h2o.H2OFrame` type.

In spite of many API similarities, `pandas.DataFrame` and `h2o.H2OFrame` classes are not related in any way, and cannot be used interchangeably in Python application code.
The former represents local data, which is stored in local computer memory as "live" NumPy arrays.
The latter represents **a reference** to remote data, which is stored in H2O.ai server memory.

An `H2OFrame` object can be constructed from, and be deconstruced back to a `DataFrame` object.
However, these operations are rather slow, because they involve transmitting the complete dataset between local and remote machines:

``` python
from h2o import H2OFrame

import h2o
import numpy
import pandas

h2o.init()

pandas_df = pandas.read_csv(...)

# Uploads to H2O.ai server
h2o_df = H2OFrame(pandas_df)

# Downloads from H2O.ai server
pandas_h2o_df = h2o_df.as_data_frame()

print("All values equal after round-trip: {}".format(numpy.all(pandas_df == pandas_h2o_df)))
```

**Important**: All operations that interact with an `H2OFrame` object (whether directly or indirectly) require an active connection to an H2O.ai server.

Getting the standard Scikit-Learn pipeline to fit:

``` python
pipeline = Pipeline([
  ("classifier", H2OGeneralizedLinearEstimator())
])

h2o_X = H2OFrame(X)
h2o_y = H2OFrame(y.to_frame(), column_types = ["categorical"])

pipeline.fit(h2o_X, h2o_y)
```

### Semi-automated data upload to H2O.ai server

If the pipeline contains any transformer steps, then the manual data upload approach will not work, because Scikit-Learn transformer classes do not support data container types other than `numpy.ndarray` and `pandas.DataFrame`.

In practice, most transformers interact with the `X` argument of the `TransformerMixin.fit_transform(X, y)` method, but not with the `y` argument.
It means that the `X` dataset cannot be uploaded until the last transformer step has completed, whereas the `y` dataset can be uploaded right away.

The `h2o` package does not provide a meta-transformer class for data uploading.
This gap is filled by the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package in the form of the `sklearn2pmml.preprocessing.h2o.H2OFrameConstructor` class.

The best place for inserting a data upload step is right before the final H2O estimator step:

``` python
from sklearn2pmml.preprocessing.h2o import H2OFrameConstructor

pipeline = Pipeline([
  ("transformer", ...),
  ("uploader", H2OFrameConstructor()),
  ("classifier", H2OGeneralizedLinearEstimator())
])
pipeline.fit(X, H2OFrame(y.to_frame(), column_types = ["categorical"]))
```

### H2O.ai pipeline persistence

Scikit-Learn developers recommend using Python's built-in pickle data format for short-term persistence needs:

``` python
import pickle

with open("pipeline.pkl", "wb") as f:
  pickle.dump(pipeline, f)
```

Unfitted `H2OEstimator` objects can be pickled and unpickled freely.

However, any attempt to pickle a fitted `H2OEstimator` object shall fail with the following pickling error in latest H2O.ai versions (at the time of writing this (November 2022), eg. 3.34.0.8, 3.36.1.5, 3.38.0.2):

```
Traceback (most recent call last):
  File "main.py", line 4, in <module>
    pickle.dump(pipeline, f)
_pickle.PicklingError: Can't pickle <class 'h2o.estimators.glm.H2OGeneralizedLinearEstimator'>: it's not the same object as h2o.estimators.glm.H2OGeneralizedLinearEstimator
```

The pickling will work if the H2O.ai version is downgraded to 3.32.1.7 or older.

### Resources

* Python script: [`train.py`]({{ "/resources/2022-11-11/train.py" | absolute_url }})