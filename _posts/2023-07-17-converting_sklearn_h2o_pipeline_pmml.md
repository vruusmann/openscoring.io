---
layout: post
title: "Converting Scikit-Learn H2O.ai pipelines to PMML"
author: vruusmann
keywords: h2o scikit-learn sklearn2pmml data-categorical
related_posts:
  - 2022-11-11-sklearn_h2o_pipeline
---

The [`h2o`](https://github.com/h2oai/h2o-3/tree/master/h2o-py) package provides Python language wrappers for H2O.ai estimators.
One and the same class can be used in standalone mode (ie. the train-predict API) as well as in Scikit-Learn pipeline mode (ie. the fit-predict API).

A Scikit-Learn H2O.ai pipeline must address two extra challenges, which relate to bridging the gap between the "local" Scikit-Learn/Python environment and the "remote" H2O.ai/Java environment:

1. Uploading training and testing datasets from local to remote.
2. Downloading models from remote to local.

## Training ##

A Scikit-Learn H2O.ai pipeline template:

``` python
from h2o import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.compose import ColumnTransformer
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.h2o import H2OFrameConstructor

import h2o

h2o.init()

pipeline = PMMLPipeline([
  ("initializer", ColumnTransformer(
    [(cat_col, CategoricalDomain(), [cat_col]) for cat_col in cat_cols] +
    [(cont_col, ContinuousDomain(), [cont_col]) for cont_col in cont_cols]
  )),
  ("uploader", H2OFrameConstructor()),
  ("classifier", H2ORandomForestEstimator())
])
pipeline.fit(X, H2OFrame(y.to_frame(), column_types = ["categorical"]))

h2o.shutdown()
```

The initializer step is a column mapper (meta-)transformer that captures the detailed description of the training dataset using SkLearn2PMML decorators.

The pipeline does not perform any data pre-processing transformations.
Unlike Scikit-Learn modeling algorithms, the majority of H2O.ai modeling algorithms can accept non-numeric columns as-is.

In fact, uncalled-for helper transformations may harm the predictive performance of a pipeline.
For example, H2O.ai decision tree algorithms can generate set-style categorical splits ("<value> in <set of ref values>") for string columns. However, they fall back to binary indicator-style categorical splits ("<value> is <ref value>") when the string column has been one-hot encoded into multiple integer columns.

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides the `sklearn2pmml.preprocessing.h2o.H2OFrameConstructor` (meta-)transformer for uploading datasets from within Scikit-Learn pipelines.

**Important**: The data uploader step can be inserted only into one specific location in the pipeline - right between the last Scikit-Learn transformer step and the first H2O.ai transformer (eg. PCA, TF-IDF, Word2Vec) or model step.
This stems from the fact that data upload changes the type of the `X` dataset from Pandas' data frame or Numpy array to H2O.ai data frame (ie. the `h2o.H2OFrame` type), thereby making it unacceptable to Scikit-Learn estimators (other than passthrough transformers).

The `(PMML)Pipeline.fit(X, y)` method call runs the data pre-processing part on the local computer and the model training part on the remote H2O.ai cluster - all as a single transaction.

The model is a Java object that is tightly coupled to its parent H2O.ai/Java environment.
It can be downloaded for backup/archival purposes into the local computer in Java serialization (short-term storage) or MOJO data formats (long-term storage, custom Java applications).

Downloading the model in MOJO data format:

``` python
classifier = pipeline._final_estimator

mojo_path = "/path/to/rf.mojo.zip"

classifier.download_mojo(path = mojo_path)
```

## Persistence ##

Any attempt to pickle a fitted `H2OEstimator` object shall fail with the following pickling error:

```
Traceback (most recent call last):
  File "train.py", line 50, in <module>
    joblib.dump(pipeline, pkl_file)
  ...
_pickle.PicklingError: Can't pickle <class 'h2o.estimators.random_forest.H2ORandomForestEstimator'>: it's not the same object as h2o.estimators.random_forest.H2ORandomForestEstimator
```

The technical explanation is that the Python class definition of an H2O.ai estimator gets modified during the `H2OEstimator.fit(X, y)` method call, by making it a subclass of various model extension classes.
For example, the `H2ORandomForestEstimator` class gets added to `h2o.model.extensions.VariableImportance`, `h2o.model.extensions.Contributions` and `h2o.model.extensions.Fairness` class hierarchies:

``` python
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.model.extensions import VariableImportance

classifier = H2ORandomForestEstimator()

# False before fit
assert not isinstance(classifier, VariableImportance)

classifier.fit(X, y)

# True after fit
assert isinstance(classifier, VariableImportance)
```

The workaround is to switch from pure Pickle to dill-flavoured Pickle data format:

``` python
import dill

with open("H2ORandomForestAudit.pkl", "wb") as pkl_file:
  dill.dump(pipeline, pkl_file)
```

The longevity and pervasive nature of the above pickling error suggests that this might be a deliberate restriction rather than a bug.

For reference, the H2O.ai documentation does not place a direct veto on pickling.
It advises that the only supported way of persisting fitted `H2OEstimator` objects is via a pair of [`h2o.download_model`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/h2o.html#h2o.download_model) and [`h2o.upload_model`](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/h2o.html#h2o.upload_model) utility functions:

``` python
import h2o

classifier = pipeline._final_estimator

h2o_backup_file = h2o.download_model(classifier, path = "/path/to/h2o_backup_dir")

classifier_clone = h2o.upload_model(h2o_backup_file)
```

Unfortunately, this advice falls short in the current case, as the `H2OEstimator` object is not a standalone entity, but is embedded into a much bigger, different language/application environment object.

## Conversion ##

The [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) library integrates seamlessly with other JPMML-family conversion libraries such as [JPMML-H2O](https://github.com/jpmml/jpmml-h2o), [JPMML-LightGBM](https://github.com/jpmml/jpmml-lightgbm), [JPMML-StatsModels](https://github.com/jpmml/jpmml-statsmodels) and [JPMML-XGBoost](https://github.com/jpmml/jpmml-xgboost).

The conversion of Scikit-Learn H2O.ai pipelines is a bit more complicated than other cross-ML framework pipelines because of H2O.ai's inherent "local" vs. "remote" dichotomy.

The JPMML-SkLearn converter assumes that the input Pickle file contains a pipeline object in its most complete state.
This assumption is violated in the case of H2O.ai estimators, because their fitted state holds a reference to a model in a remote H2O.ai cluster, rather than a fully-functional model itself.

The fix is to enhance the H2O.ai estimator with MOJO information.

The JPMML-SkLearn library supports two H2O.ai estimator enhancement styles.
First, smaller MOJO files can be read into an in-memory byte array, and assigned to the `_mojo_bytes` extension attribute:

``` python
classifier = pipeline._final_estimator

with open("/path/to/rf.mojo.zip", "rb") as mojo_file:
  classifier._mojo_bytes = mojo_file.read()
```

Second, bigger MOJO files should be left where they are.
The path to a backing MOJO file can be assigned to the `_mojo_path` extension attribute:

``` python
classifier = pipeline._final_estimator

classifier._mojo_path = "/path/to/rf.mojo.zip"
```

The good news is that starting from SkLearn2PMML version 0.95.0, the `sklearn2pmml.sklearn2pmml` utility function takes full care of all the above H2O.ai estimator enhancement and flavoured pickling details.

After a package update, the workflow simplifies back to the canonical one:

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

import h2o

h2o.init()

pipeline = PMMLPipeline(...)
pipeline.fit(X, y)

# The call must happen while connected to a remote H2O.ai cluster
sklearn2pmml(pipeline, "H2ORandomForestAudit.pmml")

h2o.shutdown()
```

## Resources ##

* Dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2023-07-17/train.py" | absolute_url }})