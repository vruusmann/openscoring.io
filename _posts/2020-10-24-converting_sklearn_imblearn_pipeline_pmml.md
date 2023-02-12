---
layout: post
title: "Converting Scikit-Learn Imbalanced-Learn pipelines to PMML"
author: vruusmann
keywords: scikit-learn imbalanced-learn sklearn2pmml
---

[Imbalanced-Learn](https://github.com/scikit-learn-contrib/imbalanced-learn/) is a Scikit-Learn extension package for re-sampling datasets.

Re-sampling derives a new dataset with specific properties from the original dataset.
It is commonly used in classification workflows to optimize the distribution of class labels.

Consider, for example, a binary classification problem where the ratio of "event" vs. "no-event" labels is heavily skewed and fluctuates across datasets.
Re-sampling can be used to enrich the dataset (by either over-sampling the "event" label, or under-sampling the "no-event" label) at a stable, desired level, which is crucial for repeatable and reproducible data science experiments.

The `imblearn` package provides samplers and sampling-aware classifiers.

Imbalanced-Learn samplers are similar to Scikit-Learn selectors, except they operate on data matrix rows rather than columns.
A sampler may lower the height of a data matrix by removing undesired rows, or increase it by inserting desired rows (either by duplicating existing rows or generating new rows from scratch).

There are popular ensemble classification algorithms that perform extra re-sampling as part of their "business logic".
For example, the random forest algorithm draws a unique subsample for training each member decision tree as a means to improve the predictive accuracy and control over-fitting.

Imbalanced-Learn classifiers such as `imblearn.ensemble.BalancedBaggingClassifier` and `imblearn.ensemble.BalancedRandomForestClassifier` extend Scikit-Learn classifiers with basic re-sampling functionality.

This blog post demonstrates how to incorporate Imbalanced-Learn samplers into PMML pipelines.

The "audit" dataset contains 1899 data records; 447 of them are labeled as "event" and 1452 as "no-event".
In this exercise, the dataset shall be enriched from the initial ~1/4 event ratio to 1/3 event ratio by randomly sampling 1000 "event" data records and 2000 "no-event" data records using the `imblearn.over_sampling.RandomOverSampler` sampler.

The sampler step is typically placed between feature engineering and classifier steps:

``` python
from imblearn.over_sampling import RandomOverSampler
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

cat_columns = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), OneHotEncoder()]) for cat_column in cat_columns] +
  [([cont_column], [ContinuousDomain()]) for cont_column in cont_columns]
)
sampler = RandomOverSampler(sampling_strategy = {0 : 2000, 1 : 1000})
classifier = DecisionTreeClassifier()

pipeline = Pipeline([
  ("mapper", mapper),
  ("sampler", sampler),
  ("classifier", classifier)
])
```

It should be pointed out that a sampler step creates new internal data matrices during fitting that shall live in computer memory side-by-side with incoming data matrices.
This is not a problem with the "audit" dataset, but may become an issue when working with Big Data-scale datasets.

However, any attempt to insert a sampler step directly into a Scikit-Learn pipeline fails with the following type error:

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.7/site-packages/sklearn/utils/validation.py", line 72, in inner_f
    return f(**kwargs)
  File "/usr/local/lib/python3.7/site-packages/sklearn/pipeline.py", line 114, in __init__
    self._validate_steps()
  File "/usr/local/lib/python3.7/site-packages/sklearn/pipeline.py", line 162, in _validate_steps
    "'%s' (type %s) doesn't" % (t, type(t)))
TypeError: All intermediate steps should be transformers and implement fit and transform or be the string 'passthrough' 'RandomUnderSampler(sampling_strategy={0: 1000, 1: 500})' (type <class 'imblearn.under_sampling._prototype_selection._random_under_sampler.RandomUnderSampler'>) doesn't
```

Imbalanced-Learn samplers are completely separate from Scikit-Learn transformers.
They inherit from the `imblearn.base.SamplerMixing` base class, and their API is centered around the `fit_resample(X, y)` method that operates both on feature and label data.

The `imblearn` package provides a `imblearn.pipeline.Pipeline` class, which extends the `sklearn.pipeline.Pipeline` class with support for sampler steps.

Switching pipeline implementations:

``` python
from imblearn.pipeline import Pipeline
#from sklearn.pipeline import Pipeline

imblearn_pipeline = Pipeline([
  ("mapper", mapper),
  ("sampler", sampler),
  ("classifier", classifier)
])
```

In principle, it is only the sampler step and the subsequent steps that must be "escaped" by wrapping them into the Imbalanced-Learn pipeline.
All steps preceding the sampler step may be left out of it.

Combining pipeline implementations:

``` python
import imblearn
import sklearn

pipeline = sklearn.pipeline.Pipeline([
  ("mapper", mapper),
  ("pipeline", imblearn.pipeline.Pipeline([
    ("sampler", sampler),
    ("classifier", classifier)
  ]))
])
```

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides an `sklearn2pmml.sklearn2pmml(pipeline: Pipeline, pmml_output_path: str)` utility function for converting Scikit-Learn pipelines to the Predictive Model Markup Language (PMML) representation.

This utility function refuses to accept Imbalanced-Learn pipeline objects as the first argument.
The associated type error suggests using the `sklearn2pmml.make_pmml_pipeline(obj)` utility function for transforming custom objects to a PMML pipeline object.
However, it is better to ignore this advice, and construct and fit a `PMMLPipeline` object explicitly:

``` python
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

import pandas

df = pandas.read_csv("Audit.csv")

#imblearn_pipeline.fit(df, df["Adjusted"])
#pmml_pipeline = make_pmml_pipeline(imblearn_pipeline)

pmml_pipeline = PMMLPipeline([
  ("pipeline", imblearn_pipeline)
])
pmml_pipeline.fit(df, df["Adjusted"])
#pmml_pipeline.configure(compact = False)
pmml_pipeline.verify(df.sample(frac = 0.01))

sklearn2pmml(pmml_pipeline, "ImbLearnAudit.pmml", with_repr = True)
```

Re-sampling is solely a training-time phenomenon.
Imbalanced-Learn samplers act as identity transformers during prediction. It means that they pass through testing and validation datasets unchanged.

Consequently, samplers are functionally void in the PMML representation.
The only trace left of them are differing data record counts as reported by different pipeline steps.
For example, the initial domain decorator steps (eg. `ContinuousDomain` and `CategoricalDomain` classes) report a record count of 1899, whereas the final estimator step (ie. the `DecisionTreeClassifier`  class) reports it as 3000.

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2020-10-24/train.py" | absolute_url }})