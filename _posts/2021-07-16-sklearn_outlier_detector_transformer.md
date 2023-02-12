---
layout: post
title: "Extending Scikit-Learn with outlier detector transformer"
author: vruusmann
keywords: scikit-learn scikit-lego sklearn2pmml
---

Outlier detection is a subfield of unsupervised learning, where the objective is to assign **anomaly score** to data records based on their feature values alone.
A data record is considered to be anomalous if it deviates from the average sample. For example, exhibiting extreme feature value(s), exhibiting an unusual combination of feature values, etc.

Scikit-Learn outlier detector classes inherit from the `sklearn.base.OutlierMixin` base class, which is completely separate from common estimator base classes such as `sklearn.base.(ClassifierMixin|RegressorMixin)` or `sklearn.base.TransformerMixin`.

Outlier detector classes are very similar to model classes API-wise, because the main "interaction point" is the `predict(X)` method.
The `predict(X)` method returns a bi-valued categorical integer, where the "outlier" and "inlier" categories are denoted by `-1` and `+1` values, respectively.
Some outlier detector classes may additionally define a `decision_function(X)` method, which returns the raw anomaly score as a continuous float.

The estimation of anomaly scores may be a valid end goal in and of itself:

``` python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

transformer = ColumnTransformer([
  ("cont", "passthrough", cont_columns),
  ("cat", OneHotEncoder(sparse = False), cat_columns)
])

outlier_detector = IsolationForest(random_state = 13)

pipeline = Pipeline([
  ("transformer", transformer),
  ("outlier_detector", outlier_detector)
])
pipeline.fit_predict(X, y = None)
```

### Anomaly score as a feature

A data scientist may wish to enrich the dataset with the anomaly score feature in order to train models that exhibit improved predictive performance, and are safer to deploy.
For example, implementing a row filterer based on anomaly score thresholding.

The training dataset should always be cleaned from anomalous data records to ensure that the learned model reflects a narrower and more consistent statistical hypothesis.

Validation and testing datasets may also be cleaned.
However, in some application scenarios it may be necessary to make the prediction for all data records in the dataset, but label them with an auxiliary quality information (eg. "reliable", "unsure", "not at all reliable" depending on how well the data record fits into the applicability domain of the model).

An outlier detector must be connected to a pipeline in a parallel way.
This is easily accomplished using the [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) meta-transformer, which applies a list of transformers to the same input data matrix, and then concats their results column-wise.

If the input data matrix needs to be preserved unchanged, then one of the list elements must be an identity transformer.
Unfortunately, unlike many other Scikit-Learn meta-transformers, the `FeatureUnion` meta-transformer does not support using the "passthrough" instruction.

A reusable `sklego.preprocessing.IdentityTransformer` transformer can be found from the [`scikit-lego`](https://github.com/koaning/scikit-lego) package:

Making a feature matrix enricher:

``` python
from sklearn.pipeline import FeatureUnion
from sklego.preprocessing import IdentityTransformer

enricher = FeatureUnion([
  ("identity", IdentityTransformer()),
  ("outlier_detector", outlier_detector)
])
```

The ordering of `FeatureUnion` list elements is not significant from the computational perspective.
It may be advisable to insert the identity transformer first and the outlier detector second, in order to avoid messing up the original feature indices.

Integrating the feature matrix enricher into a (supervised learning-) pipeline:

``` python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

pipeline = Pipeline([
  ("transformer", transformer),
  ("enricher", enricher),
  ("classifier", classifier)
])
pipeline.fit_predict(X, y)
```

It turns out that the above pipeline cannot be executed, because Scikit-Learn expects all estimators except for the final estimator to declare a `transform(X)` method.

Python error:

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.7/site-packages/sklearn/utils/validation.py", line 63, in inner_f
    return f(*args, **kwargs)
  File "/usr/local/lib/python3.7/site-packages/sklearn/pipeline.py", line 846, in __init__
    self._validate_transformers()
  File "/usr/local/lib/python3.7/site-packages/sklearn/pipeline.py", line 896, in _validate_transformers
    (t, type(t)))
TypeError: All estimators should implement fit and transform. 'IsolationForest(random_state=13)' (type <class 'sklearn.ensemble._iforest.IsolationForest'>) doesn't
```

The solution is to "masquerade" the outlier detector as a transformer.

This can be accomplished in two ways.
First, if the outlier detector class can be changed, then it can be made to inherit both from `OutlierMixin` and `TransformerMixin` base classes.
Most outlier detectors are very good candidates for becoming transformers, because they implement unsupervised learning algorithms that ignore the label column (ie. assume `fit(X, y = None)`).

Second, the outlier detector can be wrapped into a general-purpose "estimator-as-transformer" transformer, which connects wrapper's `transform(X, y)` method to wrappee's `predict(X)` or `decision_function(X)` method.

Quick search reveals that such classes are plentiful in existence.
It makes sense to pick the `sklego.meta.EstimatorTransformer` meta-transformer from the `scikit-lego` package again, because it is functionally adequate, and this exercise already has this package dependency.

Making a feature matrix enricher that is fully compatible with Scikit-Learn transformer API: 

``` python
from sklego.meta import EstimatorTransformer

enricher = FeatureUnion([
  ("identity", IdentityTransformer()),
  ("outlier_detector", EstimatorTransformer(outlier_detector, predict_func = "decision_function"))
])
```

If the outlier detector is operated in "labelling mode" (by invoking the `predict(X)` method that returns a bi-valued categorical integer feature, see above), then it is advisable to explicitly encode it in order to avoid accidental type system violations.

For example, linear models accept a categorical integer feature, but cast it forcibly to a continuous float feature. During training, the categorical integer feature is associated with a single beta coefficient.
Such beta coefficient lacks deeper meaning, because it suggests that the contribution and significance of outliers (ie. the `-1` category level) is the exact opposite of inliers (ie. the `+1` category level).

If the categorical integer value is one-hot encoded (similar to raw categorical features) then each category level becomes associated with a separate beta coefficient instead. More importantly, the effect of the numeric value of the categorical integer feature is eliminated.

Improving the feature matrix enricher for the "labelling mode"  use case:

``` python
from sklearn.pipeline import make_pipeline
from sklego.meta import EstimatorTransformer

enricher = FeatureUnion([
  ("identity", IdentityTransformer()),
  #("outlier_detector", EstimatorTransformer(outlier_detector, predict_func = "decision_function"))
  ("outlier_detector", make_pipeline(EstimatorTransformer(outlier_detector, predict_func = "predict"), OneHotEncoder()))
])
```

### Conditional execution of estimators

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides `sklearn2pmml.ensemble.SelectFirstClassifier` and `sklearn2pmml.ensemble.SelectFirstClassifier` ensemble models, which can be used to implement row filtering based on anomaly score.

By analogy with `ColumnTransformer`, their constructors take a list of `(name, estimator, predicate)` tuples.
A "select first" estimator partitions the incoming dataset into subsets based on which predicate (ie. a boolean expression) first evaluates to `True` value for a data record. All the actual training and prediction work is then delegated to child estimators.

Making a "select first" classifier that partitions the dataset into "outlier" and "inlier" subsets based on the raw anomaly score:

``` python
from sklearn2pmml.ensemble import SelectFirstClassifier

classifier = SelectFirstClassifier([
  ("outlier", outlier_classifier, "X[-1] <= 0"),
  ("inlier", inlier_classifier, str(True))
])
```

By convention, during row filtering, each data record is exposed as a row vector variable called `X`.

The predicate for the "outlier" subset is specified as `X[-1] <= 0`.
The left-hand side `X[-1]` reads "get the value of the last row vector element".
The extended right-hand side `<= 0` reads "test if this value is negative". This condition has been extracted from the [`IsolationForest.predict(X)`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest.predict) method, and is fairly specific to it.
When using a different outlier detector, a different comparison operator and/or a different threshold value may be more appropriate.

The predicate for the "inlier" subset is simply a `True` constant.
Its intention is to "match everything (that has not been matched by previous predicates)".

Child estimators see different rows but identical columns.
It is fine to dispatch the raw anomaly score to the "outlier" child estimator. In fact, the magnitude of this feature might be highly significant in explaining the variance between data records.

However, this feature should be withdrawn from the "inlier" child estimator, because, by definition, this subset is free from anomalous data records.

Emulating column dropping functionality using the `ColumnTransformer` meta-transformer:

``` python
def make_column_dropper(drop_cols):
  return ColumnTransformer([
    ("drop", "drop", drop_cols)
  ], remainder = "passthrough")

classifier = SelectFirstClassifier([
  ("outlier", outlier_classifier, "X[-1] <= 0"),
  ("inlier", make_pipeline(make_column_dropper([-1]), inlier_classifier), str(True))
])
```

All the presented estimators lend themselves to conversion to the PMML representation using the `sklearn2pmml` package.

Assembling the final pipeline, and converting it to a PMML document:

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
  ("transformer", transformer),
  ("enricher", enricher),
  ("classifier", classifier)
])
pipeline.fit(X, y)

sklearn2pmml(pipeline, "SelectFirstAudit.pmml")
```

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2021-07-16/train.py" | absolute_url }})