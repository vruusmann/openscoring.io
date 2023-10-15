---
layout: post
title: "Converting Scikit-Learn PyCaret 3 pipelines to PMML"
author: vruusmann
keywords: scikit-learn pycaret sklearn2pmml data-categorical data-missing automl
---

[PyCaret](https://github.com/pycaret/pycaret) is another AutoML tool, which specializes in tabular and time-series data analyses.

The recent PyCaret upgrade from 2(.3) to 3(.0) is exciting for two reasons.
First, heavily refactored experiment API (see [PyCaret-2271](https://github.com/pycaret/pycaret/pull/2271)). The OOP-style experiment setup and command interface is much more relatable than the functional programming-style interface.
Second, replacing the majority of custom transformer classes with their Scikit-Learn equivalents, which improves interoperability with existing Scikit-Learn oriented tooling.

## PyCaret 3 fundamentals ##

Typical supervised learning workflow:

``` python
from pycaret.classification import ClassificationExperiment, RegressionExperiment

exp = ClassificationExperiment()
exp.setup(data = df, target = "y", ...)

model = exp.create_model(...)

pycaret_pipeline = exp.finalize_model(dt)

# The training dataset, excluding the target column
X = df[df.columns.drop("y")]

yt = pycaret_pipeline.predict(X)
yt_proba = pycaret_pipeline.predict_proba(X)
```

Main stages:

1. The `exp.setup(...)` method constructs a data pre-processing pipeline that meets the user-supplied requirements, and fits it with the training dataset.
2. The subsequent `exp.create_model(...)` method fits a model based on the pre-processed training dataset. The model can be further refined using `exp.tune_model(...)`, `exp.calibrate_model(...)`, etc. methods.
3. The `exp.finalize_model(...)` method distills all the relevant steps into a unitary, deployment-ready PyCaret pipeline.

The PyCaret pipeline class inherits from the Scikit-Learn pipeline class.
It overrides all fit, transform and predict methods to enable caching, which is critical when performing the same computation many times.

PyCaret 3 relies on its own data management and data flow logic, which assumes `pandas.DataFrame` as the data matrix type, and where individual columns are identified by name, not by position.

The implementation is built around the `pycaret.internal.preprocess.transformers.TransformerWrapper` meta-transformer class.
A `TransformerWrapper` object selects input column(s) from a data matrix, feeds them to the wrapped transformer object, and inserts the result column(s) back into the data matrix.

As the module name indicates, the `TransformerWrapper` meta-transformer class belongs to be PyCaret internal API.
It is nice to be aware of its existence and main behavioural characteristics, but there is no reason to import it into everyday Python scripts.

When iterating over the steps of a PyCaret pipeline, then one will see a flat sequence of `TransformerWrapper` steps, followed by the final estimator step.
There are no other transformer classes visible at the top level.

``` python
from pycaret.internal.preprocess.transformers import TransformerWrapper
from pycaret.internal.pipeline import Pipeline as PyCaretPipeline
from sklearn2pmml.util import fqn

def print_pycaret_pipeline(pipeline):
  if not isinstance(pipeline, PyCaretPipeline):
    raise TypeError()
  steps = pipeline.steps
  transformer_steps = steps[:-1]
  final_estimator_step = steps[-1]

  for transformer_step in transformer_steps:
    name = transformer_step[0]
    transformer = transformer_step[1]
    if not isinstance(transformer, TransformerWrapper):
      raise TypeErrpr()
    print("{} -> {} // {} inputs".format(name, fqn(transformer.transformer), len(transformer._include)))

  name = final_estimator_step[0]
  final_estimator = final_estimator_step[1]
  print("{} -> {}".format(name, fqn(final_estimator)))
```

PyCaret 3 performs data pre-processing in the following stages:

| Stage | Selection | Setup options |
|-------|-----------|---------------|
| Imputation | Subset | `imputation_type` (`categorical_imputation`, `numeric_imputation`) |
| Categorical-to-categorical transformations | Subset | `rare_to_value` (`rare_value`) |
| Categorical-to-numeric transformations | Subset | `encoding_method`, `max_encoding_ohe` |
| Feature generation | Full set | `polynomial_features` (`polynomial_degree`) |
| Generalization, redundant feature elimination | Full set | `low_variance_threshold`, `remove_multicollinearity` (`multicollinearity_threshold`) |
| Numeric-to-numeric transformations | Full set | `transformation` (`transformation_method`), `normalize` (`normalize_method`), `pca` (`pca_method`, `pca_components`) |
| Feature selection | Full set | `feature_selection` (`feature_selection_method`, `feature_selection_estimator`, `n_features_to_select`) |

PyCaret pipeline keeps track of column operational type (ie. ordinal, categorical, continuous aka numeric).
This information is used in the opening stages, where different column groups are subjected to different transformations. For example, splitting columns between categorical imputation vs. numeric imputation.
However, this kind of specificity disappears as soon as all data matrix columns have become numeric.

PyCaret 3 checks all the boxes of a good AutoML tool checklist:

1. Can handle diverse data science tasks.
2. Knows about the major bits of functionality available in the underlying ML framework, can parameterize and order them correctly.
3. Can perform large-scale optimizations, statistical tests.
4. Does not do anything stupid.

The key to success is the third point - if you can work really-really hard, you do not need to work that smart after all.

It is currently rather difficult to incorporate expert or domain knowledge into data pre-processing.
The manual construction and insertion of extra `TransformerWrapper` steps into PyCaret pipelines seems technically possible, but there is no offical guidance or endorsement for doing so.

A seasoned data scientist may therefore prefer to keep the most creative part of the workflow for herself, and task PyCaret with more mundane parts.

Substituting a custom data pre-processing pipeline into the workflow:

``` python
from pycaret.classification import ClassificationExperiment, RegressionExperiment

exp = ClassificationExperiment()
exp.setup(
  data = df, target = "y",
  preprocess = False, custom_pipeline = Pipeline([...])
)

model = exp.create_model(...)

pycaret_pipeline = exp.finalize_model(model)
```

## Classification experiment ##

The "audit-NA" dataset contains three numeric columns and five categorical string columns. Roughly 25% of values are missing.

The goal of the experiment is to train a binary logistic regression classifier.
A good data pre-processing pipeline would therefore need to perform imputation, scaling and some sort of redundant feature elimination.

``` python
from pycaret.classification import ClassificationExperiment

import pandas

df = pandas.read_csv("audit-NA.csv")
df = df.drop(columns = ["Deductions"], axis = 1)

print(df.dtypes)

exp = ClassificationExperiment()
exp.setup(
  data = df, target = "Adjusted",
  imputation_type = "simple",
  rare_to_value = 0.02, rare_value = "(Other)",
  encoding_method = None, max_encoding_ohe = 7,
  fix_imbalance = True,
  normalize = "zscore",
  remove_multicollinearity = True, multicollinearity_threshold = 0.75
)

model = exp.create_model(estimator = "lr")

pycaret_pipeline = exp.finalize_model(model)
print_pycaret_pipeline(pycaret_pipeline)
```

Summary of the `pycaret_pipeline` object:

| Name | Class | Columns |
|------|-------|---------|
| `numerical_imputer` | `sklearn.impute.SimpleImputer` | 3 |
| `categorical_imputer` | `sklearn.impute.SimpleImputer` | 5 |
| `rare_category_grouping` | `pycaret.internal.preprocess.transformers.RareCategoryGrouping` | 5 |
| `ordinal_encoding` | `category_encoders.ordinal.OrdinalEncoder` | 1 |
| `onehot_encoding` | `category_encoders.one_hot.OneHotEncoder` | 2 |
| `rest_encoding` | `category_encoders.leave_one_out.LeaveOneOutEncoder` | 2 |
| `remove_multicollinearity` | `pycaret.internal.preprocess.transformers.RemoveMulticollinearity` | 19 |
| `balance` | `pycaret.internal.preprocess.transformers.FixImbalancer` | 19 |
| `normalize` | `sklearn.preprocessing.StandardScaler` | 19 |
| `actual_estimator` | `sklearn.linear_model.LogisticRegression` | (all) |

The results reveal that the current configuration failed to achieve redundant feature elimination, because the `RemoveMulticollinearity` transformer draws 19 columns as input, and returns the same 19 columns as result.

## Regression experiment ##

The "auto" dataset contains four numeric columns and three numeric-like columns. The latter are manually cast to categorical string columns.

The goal of the experiment is to explore model space via AutoML means.
Data pre-processing is performed conservatively, in order to ensure a level playing field to all major algorithm families (linear models, decision trees, etc.).

During AutoML search, there are three regressor types excluded.
The first two (`catboost` and `gpc`) are excluded for technical reasons (not supported by the chosen PMML conversion software). The third one (`knn`) is excluded just to keep tension high, as k-nearest neighbors is known to perform extremely well with small and homogeneous datasets.

``` python
from pycaret.regression import RegressionExperiment

import pandas

df = pandas.read_csv("auto.csv")

cat_cols = ["cylinders", "model_year", "origin"]
for cat_col in cat_cols:
  df[cat_col] = df[cat_col].astype(str)

print(df.dtypes)

exp = RegressionExperiment()
exp.setup(
  data = df, target = "mpg",
  # Model composition changes, when omitting this attribute
  categorical_features = cat_cols,
  imputation_type = None,
  encoding_method = None, max_encoding_ohe = 3,
  normalize = True, normalize_method = "robust",
  remove_multicollinearity = True, multicollinearity_threshold = 0.9
)

# Generate models
top3_models = exp.compare_models(exclude = ["catboost", "gpc", "knn"], n_select = 3)

# Select the best model from generated models
automl_model = exp.automl(optimize = "MAE")

pycaret_pipeline = exp.finalize_model(automl_model)
print_pycaret_pipeline(pycaret_pipeline)
```

Summary of the `pycaret_pipeline` object:

| Name | Class | Columns |
|------|-------|---------|
| `onehot_encoding` | `category_encoders.one_hot.OneHotEncoder` | 1 |
| `rest_encoding` | `category_encoders.leave_one_out.LeaveOneOutEncoder` | 2 |
| `remove_multicollinearity` | `pycaret.internal.preprocess.transformers.RemoveMulticollinearity` | 9 |
| `normalize` | `sklearn.preprocessing.RobustScaler` | 8 |
| `actual_estimator` | `sklearn.linear_model.HuberRegressor` | (all) |

Surprisingly enough, this competition is won by a linear model, well ahead of various decision tree ensemble models.

## PMML ##

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides the `sklearn2pmml.sklearn2pmml` utility function for converting Scikit-Learn pipelines to the Predictive Model Markup Language (PMML) representation.

However, the `sklearn2pmml` utility function refuses to accept Python classes other than the `sklearn2pmml.pipeline.PMMLPipeline` class.

The solution is to wrap the PyCaret pipeline object into a `PMMLPipeline` object using the `sklearn2pmml.pycaret.make_pmml_pipeline` utility function.
This utility function differs from the generic `sklearn2pmml.make_pmml_pipeline` utility function by the fact that it knows about the `TransformerWrapper` meta-transformer class, and can perform proper escaping of its contents.

The escaping is needed to ensure that the "business state" of all transformers and estimators is complete when dumped in pickle data format.
The list of known troublemakers contains mostly Scikit-Learn selector classes (caused by the dynamic implementation of the `_get_support_mask()` method).
If the escaping is not done, then the conversion succeeds with simpler pipelines, but may fail with more complex ones.

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pycaret import make_pmml_pipeline as pycaret_make_pmml_pipeline

pmml_pipeline = pycaret_make_pmml_pipeline(pycaret_pipeline, target_fields = ["y"])

sklearn2pmml(pmml_pipeline, "PyCaretPipeline.pmml")
```

The `make_pmml_pipeline` utility function takes `active_fields` and `target_fields` arguments, which capture feature names and label name(s), respectively.
They can be left to their default `None` values if the column names of the training dataset are fine.

PMML documents are concise yet informative, and fully self-contained.
In the two experiments above, if the PyCaret pipeline object is first saved in PMML data format (uncompressed text) and then in pickle data format (compressed binary), then the size of the PMML file is actually smaller in both cases!

## Resources ##

* Datasets: [`audit-NA.csv`]({{ "/resources/data/audit-NA.csv" | absolute_url }}) and [`auto.csv`]({{ "/resources/data/auto.csv" | absolute_url }})
* Python scripts: [`train-classification.py`]({{ "/resources/2023-01-12/train-classification.py" | absolute_url }}) and [`train-regression.py`]({{ "/resources/2023-01-12/train-regression.py" | absolute_url }})
