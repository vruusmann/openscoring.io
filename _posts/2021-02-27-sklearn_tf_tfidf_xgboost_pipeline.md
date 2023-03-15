---
layout: post
title: "Training Scikit-Learn TF(-IDF) plus XGBoost pipelines"
author: vruusmann
keywords: scikit-learn xgboost mlxtend sklearn2pmml tf-idf data-missing
---

## You have been doing it wrong ##

Consider the simplest TF(-IDF) plus XGBoost pipeline:

``` python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

pipeline = Pipeline([
  ("countvectorizer", CountVectorizer()),
  ("classifier", XGBClassifier(random_state = 13))
])
```

**Is this pipeline correct or not**?

The question is not about spotting a typo, or optimizing the default parameterization.
The question is "are you allowed to pass the document-term matrix of `CountVectorizer` (or any of its subclasses such as `TfidfVectorizer`) directly to `XGBClassifier`, or not?".

This pipeline can be fitted without any errors or warnings, and appears to make sensible predictions.
Therefore, the anwser must be "yes", right?

Not so fast! The executability proves technical compatibility, but it does not prove logical compatibility.

Despite adhering to the standard Scikit-Learn API, these two pipeline steps both exhibit slightly non-standard behaviour. 
First, the `transform(X)` method of Scikit-Learn TF(-IDF) transformers produces sparse not dense data matrices.
For example, the "sentiment" dataset is expanded into a compressed sparse row (CSR) `scipy.sparse.csr.csr_matrix` data matrix of shape `(1000, 1847)`, which has ~0.005 density (ie. only 0.5% of cells hold non-zero values).
Second, the `fit(X, y)` and `predict(X)` methods of XGBoost estimators accept most common [SciPy](https://scipy.org/), [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) data structures.
However, behind the scenes, they are all converted to a proprietary `xgboost.DMatrix` data matrix.

It is possible to reduce complexity in the contact area by explicitly converting the document-term matrix from sparse to dense representation.

The `CountVectorizer` transformer does not provide any controls (eg. a "sparse" constructor parameter) for that.
A good workaround is to use the `mlxtend.preprocessing.DenseTransformer` pseudo-transformer from the [`mlxtend`](https://github.com/rasbt/mlxtend) package:

``` python
from mlxtend.preprocessing import DenseTransformer

pipeline = Pipeline([
  ("countvectorizer", CountVectorizer()),
  ("densifier", DenseTransformer()),
  ("classifier", XGBClassifier(random_state = 13))
])
```

This pipeline (dense) should be functionally identical to the first one (sparse), but somehow it is making different predictions!
For example, the predicted probabilities for the first data record of the "sentiment" dataset are `[0.9592051, 0.04079489]` and `[0.976111, 0.02388901]`, respectively.

Clearly, one of the two pipelines must be incorrect.

## Untangling the mess ##

The situation cannot be definitively cleared up by making more predictions, or exploring the documentation and Python source code of relevant classes and methods.

Converting the pipeline to the Predictive Model Markup Language (PMML) representation, and making predictions using a PMML engine provides an objective (ie. first-principles) second opinion.

Converting using the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package:

``` python
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

pipeline = Pipeline(...)

pmml_pipeline = make_pmml_pipeline(pipeline, active_fields = ["Sentence"], target_fields = ["Score"])

sklearn2pmml(pmml_pipeline, "XGBSentiment.pmml")
```

In the current case, it does not matter which pipeline of the two is converted.
The resulting PMML documents will be identical (except for the conversion timestamp in the header), because the `DenseTransformer` pseudo-transformation is no-op.

Making predictions using the [`jpmml_evaluator`](https://github.com/jpmml/jpmml-evaluator-python) package:

``` python
from jpmml_evaluator import make_evaluator

import pandas

evaluator = make_evaluator("XGBSentiment.pmml") \
  .verify()

arguments = pandas.read_csv("sentiment.csv")

results = evaluator.evaluateAll(arguments)
print(results.head(5))
```

PMML predictions are in perfect agreement with the predictions of the second pipeline (dense).

**It follows that the first pipeline (sparse) is indeed incorrect**. 

The source of the error is the algorithm that the XGBoost library uses for converting `scipy.sparse.csr.csr_matrix` to `xgboost.DMatrix`.

The document-term matrix keeps count how many times each document (rows) contains each term (columns).
The cell value is set only if the count is greater than zero.
The DMatrix converter appears to interpret unset cell values as missing values (`NaN`) rather than zero count values (`0`).
In plain english, these interpretations read like "I do not know if the document contains the specified term" and "I know that the document contains zero occurrences of the specified term", respectively.

Scikit-Learn estimators typically error out when they encounter `NaN` values.
In contrast, XGBoost estimators treat `NaN` values as special-purpose missing value indicator values, and grow missing value-aware decision trees.

When comparing XGBoost estimators between the first and the second pipeline, then they are structurally different (overall vocabulary, the time and location of individual term invocations, etc.).
The former incorrectly believes that it was dealing with massive amounts of missing values during training, and all its internals are thus systematically off.

A data scientist may evaluate such a biased TF(-IDF) plus XGBoost pipeline with a validation dataset, and decide that its raw numeric performance is still good enough for productionization.
It would be okay. The Pipeline API provides adequate guarantees that all biases survive and are consistently applied throughout the pipeline life-cycle.

## Doing it right ##

As of XGBoost 1.3(.3), the `missing` constructor parameter has no effect:

``` python
pipeline = Pipeline([
  ("countvectorizer", CountVectorizer()),
  # Map missing value indicator value to -1 in the hope that this will change the interpretation of unset cell values from missing values to zero count values
  ("classifier", XGBClassifier(mising = -1.0, random_state = 13))
])
# Raises a UserWarning: "`missing` is not used for current input data type:<class 'scipy.sparse.csr.csr_matrix'> str(type(data)))"
pipeline.fit(df["Sentence"], df["Score"])
```

This user warning should be taken seriously, and the fitted pipeline abandoned, because it is incorrect again.

Converting the document-term matrix from sparse to dense representation is good for quick troubleshooting purposes.
However, it is prohibitively expensive in real-life situations where the dimensions of data matrices easily reach millions of rows (documents) and/or tens of thousands of columns (terms).

The solution is to perform the conversion from `scipy.sparse.csr.csr_matrix` to `xgboost.DMatrix` over a temporary sparse `pandas.DataFrame`:

``` python
X = df["Sentence"]
y = df["Score"]

countvectorizer = CountVectorizer()
Xt = countvectorizer.fit_transform(X)

# Convert from csr_matrix to sparse DataFrame
Xt_df_sparse = DataFrame.sparse.from_spmatrix(Xt)
print("DF density: {}".format(Xt_df_sparse.sparse.density))

classifier = XGBClassifier(random_state = 13)
classifier.fit(Xt_df_sparse, y)

yt_proba = classifier.predict_proba(Xt_df_sparse)
```

The above TF(-IDF) plus XGBoost sequence is correct in a sense that unset cell values are interpreted as zero count values.

The only problem is that this sequence cannot be "formatted" as a `Pipeline` object, because there is no reusable (pseudo-)transformer that would implement the intermediate `DataFrame.sparse.from_spmatrix(data)` method call.

However, fitted pipeline steps can be combined into a temporary pipeline for PMML conversion purposes:

``` python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(countvectorizer, classifier)

pmml_pipeline = make_pmml_pipeline(pipeline, active_fields = ["Sentence"], target_fields = ["Score"])

sklearn2pmml(pmml_pipeline, "XGBSentiment.pmml")
```

## Resources ##

* "Sentiment" dataset: [`sentiment.csv`]({{ "/resources/data/sentiment.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2021-02-27/train.py" | absolute_url }})
