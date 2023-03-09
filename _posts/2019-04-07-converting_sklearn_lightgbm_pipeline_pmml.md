---
layout: post
title: "Converting Scikit-Learn LightGBM pipelines to PMML"
author: vruusmann
keywords: scikit-learn lightgbm sklearn2pmml data-categorical data-missing
---

[LightGBM](https://github.com/Microsoft/LightGBM) is a serious contender for the top spot among gradient boosted trees (GBT) algorithms.

Even though it can be used as a standalone tool, it is mostly used as a plugin to more sophisticated ML frameworks such as Scikit-Learn or R.
The idea is to use the underlying ML framework for generic activities such as loading, cleaning and preparing data, and use a third-party library only in the final stage.
As a specialized library, LightGBM offers much better performance (eg. distributed and hardware-accelerated backends) and richer parameterization options.

Getting started with third-party libraries is fairly easy on Scikit-Learn, because everything is organized around the pipeline concept, and the roles and responsibilities of individual pipeline steps are formalized via an API.
For example, a Scikit-Learn pipeline that was constructed around the `sklearn.ensemble.GradientBoostingClassifier` model can be upgraded to LightGBM by simply replacing it with the `lightgbm.LGBMClassifier` model.

However, in order to unlock the full potential of third-party libraries, it becomes necessary to learn about their main characteristics and assumptions by systematically going through their documentation and code examples.
It is often the case that a good portion of key functionality remains unused, because end users simply do not know about it, or cannot find a way to implement it in practice.

This blog post demonstrates how to take full advantage of LightGBM categorical feature support and missing values support.

The exercise starts with defining a generic two-step pipeline that trains a binary classification model for the "audit" dataset.
In brief, the dataset contains both categorical and continuous features, which are separated from one another and subjected to operational type-dependent data pre-processing using the `sklearn_pandas.DataFrameMapper` meta-transformer.
Categorical features are mapped one by one, by first capturing their domain using the `sklearn2pmml.decoration.CategoricalDomain` decorator and then binarizing and/or integer encoding them using Scikit-Learn's built-in label transformers.
Continuous features are mapped all together, simply by capturing their domain using the `sklearn2pmml.decoration.ContinuousDomain` decorator.

``` python
from lightgbm import LGBMClassifier
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
  [(cont_columns, ContinuousDomain())]
)
classifier = LGBMClassifier(objective = "binary", n_estimators = 31, random_state = 42)

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", classifier)
])
pipeline.fit(df, df["Adjusted"])

sklearn2pmml(pipeline, "LightGBMAudit.pmml") 
```

LightGBM models can be converted to the standardized Predictive Model Markup Language (PMML) representation using the [JPMML-LightGBM](https://github.com/jpmml/jpmml-lightgbm) library.
Just like LightGBM itself, this library can be used as a standalone tool or as a plugin to other JPMML family conversion tools and libraries.
The main difference between these two usage modes is related to the sourcing of feature definitions.
In standalone mode, they are extracted from the LightGBM model object.
In plugin mode, they are inherited from the host ML framework, and only checked for consistency against the LightGBM model object.

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides `CategoricalDomain` and `ContinuousDomain` decorators specifically for the purpose of ensuring that Scikit-Learn feature definitions are as rich and nuanced as possible.

### Categorical features

The `LabelBinarizer` transformer expands a string column to a list of integer columns, one for each category level. For example, the "Education" column is expanded to sixteen integer columns (with cell values being either 0 or 1).

The LightGBM classifier in its default configuration, just like all Scikit-Learn estimators, treats binary features as regular numeric features.
Continuous splits are encoded using the `SimplePredicate` element:

``` xml
<Node id="19" recordCount="134.0" defaultChild="-2">
  <SimplePredicate field="Hours" operator="lessOrEqual" value="41.50000000000001"/>
  <Node id="-2" score="-1.247754842205732" recordCount="109.0">
    <SimplePredicate field="Education=College" operator="lessOrEqual" value="1.0000000180025095E-35"/>
  </Node>
  <Node id="-21" score="-1.1311261704956659" recordCount="25.0">
    <SimplePredicate field="Education=College" operator="greaterThan" value="1.0000000180025095E-35"/>
  </Node>
</Node>
```

The backing binary features are defined under the `/PMML/TransformationDictionary` element:

``` xml
<DerivedField name="Education=College" optype="continuous" dataType="double">
  <NormDiscrete field="Education" value="College"/>
</DerivedField>
```

While correct, the above PMML markup is not particularly elegant.

LightGBM uses a type system, where continuous and categorical features are represented using `double` and `integer` values, respectively.
This is different from Scikit-Learn GBT algorithms, which do not use the notion of an operational type, and represent everything using `float` values.

LightGBM has categorical feature detection capabilities, but since the output of a `DataFrameMapper` step is a 2-D Numpy array of `double` values, it does not fire correctly.
The solution is to supply the indices of categorical features manually, by specifying a `categorical_feature` fit parameter to the `LGBMClassifier.fit(X, y, **fit_params)` method.
Since the LightGBM classifier is contained inside a pipeline object and the interaction is intermediated by the `Pipeline.fit(X, y, **fit_params)` method, then the name of this fit parameter needs to be prefixed with the name of the step (ie. `categorical_feature` becomes `classifier__categorical_feature`):

``` python
from sklearn.preprocessing import LabelBinarizer

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
  [(cont_columns, ContinuousDomain())]
)

# A categorical feature transforms to a variable number of binary features.
# One way of obtaining a list of binary feature indices is to transform the dataset, and exclude the indices of known continuous features
Xt = mapper.fit_transform(df)
cat_indices = [i for i in range(0, Xt.shape[1] - len(cont_columns))]

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)
```

The predictive performance of the LightGBM classifier is completely unaffected by this operational type hint.
However, the PMML document is much conciser now, because earlier continuous splits (`SimplePredicate` value comparison operators `lessOrEqual` and `greaterThan`) have been replaced with categorical splits (ie. `SimplePredicate` equality check operators `equal` and `notEqual`):

``` xml
<Node id="19" recordCount="134.0" defaultChild="-21">
  <SimplePredicate field="Hours" operator="lessOrEqual" value="41.50000000000001"/>
  <Node id="-4" score="-1.1311261704956659" recordCount="25.0">
    <SimplePredicate field="Education" operator="equal" value="College"/>
  </Node>
  <Node id="-21" score="-1.247754842205732" recordCount="109.0">
    <SimplePredicate field="Education" operator="notEqual" value="College"/>
  </Node>
</Node>
```

When a categorical feature is manually transformed to a list of binary features, then one is depriving LightGBM from seeing the categorical feature for what it actually is, and performing its own, more effective and efficient transformation work.

However, there is a minor inconvenience related to the fact that LightGBM estimators do not accept string columns directly, but expect them to be re-encoded as integer columns. For example, the "Education" column needs to be transformed to a sole integer column (with cell values ranging from 0 to 15).

The `LabelEncoder` transformer provides this exact functionality:

``` python
from sklearn.preprocessing import LabelEncoder

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), LabelEncoder()]) for cat_column in cat_columns] +
  [(cont_columns, ContinuousDomain())]
)

# A categorical string feature transforms to exactly one categorical integer feature.
# The list of categorical feature indices contains len(cat_columns) elements
cat_indices = [i for i in range(0, len(cat_columns))]

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)
```

A binary-style categorical split discriminates one category level against all others ("send category level A to the left, and all other category levels to the right").
A multinomial-style categorical split examines every category level, and proposes two distinct subsets based on their discriminatory effect ("send category levels A, C, D and F to the left, and category levels B, and E to the right").
LightGBM does not appear to limit the cardinality of categorical features. There is no need to change anything about the Python script when scaling from two to twenty thousand category levels.

Multinomial-style categorical splits are encoded using the `SimpleSetPredicate` element:

``` xml
<Node id="3" recordCount="1021.0" defaultChild="27">
  <SimpleSetPredicate field="Marital" booleanOperator="isIn">
    <Array type="string">Absent Divorced Married-spouse-absent Unmarried Widowed</Array>
  </SimpleSetPredicate>
  <Node id="4" recordCount="271.0" defaultChild="-2">
    <SimpleSetPredicate field="Education" booleanOperator="isIn">
      <Array type="string">Associate Bachelor Doctorate Master Professional Yr12 Yr9</Array>
    </SimpleSetPredicate>
    <!-- Omitted Node elements -->
  </Node>
  <Node id="27" recordCount="750.0" defaultChild="-5">
    <SimpleSetPredicate field="Education" booleanOperator="isIn">
      <Array type="string">College HSgrad Preschool Vocational Yr10 Yr11 Yr1t4 Yr5t6 Yr7t8</Array>
    </SimpleSetPredicate>
    <!-- Omitted Node elements -->
  </Node>
</Node>
```

The predictive performance of the LightGBM classifier did not change much (some metrics improved, some others deteriorated slightly).
The biggest impact is observed around the estimated feature importances instead.
When a categorical feature is binarized, then each category level is benchmarked in isolation. In contrast, when a categorical feature is integer-encoded, then category levels "stay together" and are benchmarked as an aggregate.

### Missing values

Another major advantage of LightGBM is its ability to deal with missing values (aka sparse data).

For example, when substituting the dense "audit" dataset with a sparse "audit-NA" dataset, then the `LabelEncoder` transformer is no longer able to perform its function:

``` python
df = pandas.read_csv("audit-NA.csv")

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)
```

Python error:

```
Traceback (most recent call last):
  File "/usr/lib/python3.4/site-packages/sklearn_pandas/pipeline.py", line 24, in _call_fit
    return fit_method(X, y, **kwargs)
  File "/usr/lib/python3.4/site-packages/sklearn_pandas/pipeline.py", line 84, in fit_transform
    Xt, y, **fit_params)
  File "/usr/lib/python3.4/site-packages/sklearn_pandas/pipeline.py", line 27, in _call_fit
    return fit_method(X, **kwargs)
  File "/usr/lib64/python3.4/site-packages/sklearn/preprocessing/label.py", line 236, in fit_transform
    self.classes_, y = _encode(y, encode=True)
  File "/usr/lib64/python3.4/site-packages/sklearn/preprocessing/label.py", line 108, in _encode
    return _encode_python(values, uniques, encode)
  File "/usr/lib64/python3.4/site-packages/sklearn/preprocessing/label.py", line 63, in _encode_python
    uniques = sorted(set(values))
TypeError: unorderable types: str() < float()
```

If data sparsity is not too high, then it can be made whole by imputing missing values based on available evidence.
For continuous features, the replacement value is typically the mean or median. For categorical features, the replacement value is typically the mode or some predefined constant (eg. "N/A").

The `SimpleImputer` transformer changes a sparse dataset to dense dataset:

``` python
from sklearn.impute import SimpleImputer

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), SimpleImputer(strategy = "most_frequent"), LabelEncoder()]) for cat_column in cat_columns] +
  [(cont_columns, ContinuousDomain())]
)

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)
```

Missing value imputation skews the training dataset towards the "average" sample, and may lead the ML algorithm to discover all sorts of biased or outright false relationships.

The inspection of the above pipeline shows that all components except the `LabelEncoder` transformer can more or less cope with missing values.
For example, both `CategoricalDomain` and `ContinuousDomain` decorators count the number of missing values in a column (for summary/statistics purposes), but otherwise simply pass them on to the next component.

It really is a poor proposition to perform the missing value imputation and risk the data science experiment just to satisfy one component that performs a helper function. Therefore, it needs to go.

The `sklearn2pmml` package provides a `sklearn2pmml.preprocessing.PMMLLabelEncoder` transformer, which is essentially a missing value-aware replacement for the `LabelEncoder` transformer.

``` python
from sklearn2pmml.preprocessing import PMMLLabelEncoder

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), PMMLLabelEncoder()]) for cat_column in cat_columns] +
  [(cont_columns, ContinuousDomain())]
)

pipeline = PMMLPipeline([...])
pipeline.fit(df, df["Adjusted"], classifier__categorical_feature = cat_indices)
```

In this final configuration, the `DataFrameMapper` step is transforming the training dataset to a dense 2-D Numpy array of `object` values, where missing continuous and categorical values are denoted by `NaN` and `None` values, respectively.
A LightGBM estimator has no problem accepting and interpreting such data matrix if the `categorical_feature` fit parameter is specified.

The predictive performance of the LightGBM classifier improves considerably across all tracked metrics (accuracy, precision, recall, ROC AUC), when the missing value-ignorant `[SimpleImputer(), LabelEncoder()]` component is replaced with the missing value-aware `PMMLLabelEncoder` component.

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* "Audit-NA" dataset: [`audit-NA.csv`]({{ "/resources/data/audit-NA.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2019-04-07/train.py" | absolute_url }})