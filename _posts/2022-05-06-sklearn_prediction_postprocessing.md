---
layout: post
title: "Extending Scikit-Learn with prediction post-processing"
author: vruusmann
keywords: scikit-learn sklearn2pmml
---

The centerpiece of ML pipelines is the model.
Steps that precede the model are called "data pre-processing" aka "feature engineering" steps. Steps that follow the model are called "prediction post-processing" aka "decision engineering" steps.

Overall, data pre-processing is more appreciated and valued than prediction post-processing.
Feature transformations allow the data to be (re)presented in more nuanced and relevant ways, thereby leading to better models.
However, even the best model is functionally compromised if its predictions are confusing or difficult to integrate with the host application.

Scikit-Learn pipelines provide data pre-processing support, but completely lack prediction post-processing support. Any attempt to insert a transformer step after the final estimator step shall fail with an error.

Potential workarounds include wrapping an estimator into a post-processing meta-estimator (that overrides the `predict(X)` method), or performing post-processing computations outside of the Scikit-Learn pipeline using free-form Python code.

The selection and functionality of meta-estimators is rather limited. The two notable examples are [`TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) for transforming the target during regression, and [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) for transforming the decision function during classification.

Performing computations using free-form Python code is the nuclear option. It allows reaching any goal, but sacrifices the main value proposition of ML pipelines, which is atomicity and ease of deployment across time and space.

## PMMLPipeline "transformed prediction" API ##

The `sklearn2pmml` package provides the `sklearn2pmml.pipeline.PMMLPipeline` class, which extends the `sklearn.pipeline.Pipeline` class with prediction post-processing.

The idea is to attach a number of child transformers to the parent pipeline, one for each predict method:

| Attribute | Predict method | Transformed predict method |
|---|---|---|
| `predict_transformer` | `predict(X)` | `predict_transform(X)` |
| `predict_proba_transformer` | `predict_proba(X)` | `predict_proba_transform(X)` |
| `apply_transformer` | `apply(X)` | `apply_transform(X)` |

A transformed predict method extends the pipeline towards a particular objective.
Its output is a 2-D Numpy array, where the leftmost column(s) correspond to the primary result, and all the subsequent columns to secondary results:

``` python
import numpy

def predict_transform(X):
  yt = self.predict(X)
  yt_postproc = self.predict_transformer.transform(yt)
  return numpy.hstack((yt, yt_postproc))
```

Child transformers cannot see the incoming `X` dataset.
A data matrix may expand or contract during data pre-processing in unforeseen ways, so it would be very difficult to match a specific feature column or condition during prediction post-processing.
If a business decision is a function of both model input and output, then it still needs to be coded manually.

There is no limit to child transformer's complexity, except that it cannot encapsulate a full-blown model.

Additionally, the `sklearn2pmml` package provides the `sklearn2pmml.postprocessing.BusinessDecisionTransformer` transformer, which generates rich `OutputField` elements following the ["decision" result feature](https://dmg.org/pmml/v4-4-1/Output.html#xsdElement_Decisions) conventions.

## Examples ##

The class label of the "audit" dataset is encoded as a binary integer, where the "0" value and the "1" value indicate non-productive and productive audits, respectively.
Such internal encodings should be unwound before reaching higher application levels.

Post-processing class labels:

``` python
from sklearn2pmml.decoration import Alias
from sklearn2pmml.postprocessing import BusinessDecisionTransformer
from sklearn2pmml.preprocessing import ExpressionTransformer

binary_decisions = [
  ("yes", "Auditing is needed"),
  ("no", "Auditing is not needed")
]

pipeline = PMMLPipeline([...]
, predict_transformer = Alias(BusinessDecisionTransformer(ExpressionTransformer("'yes' if X[0] == 1 else 'no'"), "Is auditing necessary?", binary_decisions, prefit = True), "binary decision", prefit = True))

yt = pipeline.predict_transform(X)
```

Regression results can be transformed numerically using [`FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) or `ExpressionTransformer` transformers, whereas classification results can be re-mapped using the `LookupTransformer` transformer.

The `BusinessDecisionTransformer` transformer is applicable to categorical results (classification and clustering results, bucketized regression results).
It articulates the business problem, and enumerates the full range of business decisions that this output field can make.

Post-processing probability distributions:

``` python
from sklearn2pmml.decoration import Alias
from sklearn2pmml.postprocessing import BusinessDecisionTransformer
from sklearn2pmml.preprocessing import CutTransformer, ExpressionTransformer

graded_decisions = [
  ("no", "Auditing is not needed"),
  ("no over yes", "Audit in last order"),
  ("yes over no", "Audit in first order"),
  ("yes", "Auditing is needed"),
]

event_proba_quantiles = [0.0, 0.1363, 0.5238, 0.7826, 1.0]

predict_proba_transformer = Pipeline([
  ("selector", ExpressionTransformer("X[1]")),
  ("decider", Alias(BusinessDecisionTransformer(CutTransformer(bins = event_proba_quantiles, labels = [key for key, value in graded_decisions]), "Is auditing necessary?", graded_decisions, prefit = True), "graded decision", prefit = True))
])

pipeline = PMMLPipeline([...]
, predict_proba_transformer = predict_proba_transformer)

yt = pipeline.predict_proba_transform(X)
```

The input to the `predict_proba_transformer` is a multi-column array.
Therefore, the transformation is typically implemented as a pipeline, where the first step performs column selection.

In case of elementary operations it is possible to keep the transformer as a standalone pipeline step, or embed it into the `BusinessDecisionTransformer` transformer. The former approach gives rise to an extra `OutputField` element, which may be seen as an unnecessary clutter to a model schema.

The two above examples are about fully-decoupled child transformers. They are composed of prefitted components, and may be defined and assigned in the `PMMLPipeline` constructor.

However, there are several application areas where the child transformer needs to reference the internal state of the preceding estimator, or even be fitted relative to it (eg. probability calibration).
This does not pose any problems, because all the relevant `PMMLPipeline` attributes may be assigned and re-assigned at any later time.

Post-processing leaf indices:

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import Alias
from sklearn2pmml.preprocessing import LookupTransformer

classifier = DecisionTreeClassifier()

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", classifier)
])
pipeline.fit(X, y)

def leaf_sizes(tree):
  leaf_sizes = dict()
  for i in range(tree.node_count):
    if (tree.children_left[i] == -1) and (tree.children_right[i] == -1):
      leaf_sizes[i] = int(tree.n_node_samples[i])
  return leaf_sizes

pipeline.apply_transformer = Alias(LookupTransformer(leaf_sizes(classifier.tree_), default_value = -1), "leaf size", prefit = True)

yt = pipeline.apply_transform(X)

pipeline.configure(compact = False, flat = False, numeric = False, winner_id = True)
sklearn2pmml(pipeline, "DecisionTreeAudit.pmml")
```

The input to the `apply_transformer` is a column vector for decision tree models, and a 2-D Numpy array for decision tree ensemble models.

Scikit-Learn identifies decision tree nodes by 1-based integer index, which can be encoded using the PMML entity identifiers mechanism.

By default, the `sklearn2pmml` package does not collect and encode node identifiers, because that would prevent it from compacting and flattening the tree data structure.
The default behaviour is suppressed by deactivating `compact` and `flat` conversion options, and activating the `winner_id` conversion option. The `numeric` conversion option controls the encoding of categorical splits, and can toggled freely.

## Resources ##

* Dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python scripts: [`train.py`]({{ "/resources/2022-05-06/train.py" | absolute_url }}) and [`predict.py`]({{ "/resources/2022-05-06/predict.py" | absolute_url }})
