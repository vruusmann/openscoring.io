---
layout: post
title: "Extending Scikit-Learn with business rules model"
author: vruusmann
keywords: scikit-learn sklearn2pmml business-rules
---

## Machine Learning vs. Business Rules

Predictive analytics applications can be equally well served by algorithmically learned "machine learning" models or manually crafted "business rules" models.
In fact, some of the most robust and successful models are hybrids between the two, where machine generated candidates are filtered, reviewed and refactored by a human domain expert.

Consider, for example, the classification of iris species.
Different ML frameworks and classification algorithms are able to propose a large number of solutions. A human might then compare and constrast them, and distill the final solution into a simple and elegant decision tree:

``` java
if(Petal_Length < 2.45){
  return "setosa";
} else {
  if(Petal_Width < 1.75){
    return "versicolor";
  } else {
    return "virginica";
  }
}
```

## ML framework perspective

Popular ML frameworks such as R, Scikit-Learn and Apache Spark disregard the "business rules" use case.
The solution is to develop a custom model type, which has the following behaviour:

1. The constructor takes user-specified business rules.
2. The `fit(X, y)` method is no-op. As the business rules were specified once and for all during construction, any attempt to modify or re-learn them would result in indeterminate behaviour.
3. The `predict(X)` method applies user-specified business rules.

The main technical challenge is related to the representation of business rules.
First and foremost, they should be easy for the data scientist to compose and maintain. Secondly, they should be easy to implement with the underlying ML framework.

There are many third-party business rules solutions available for the Python platform. After some tinkering they were all pushed aside in favour of Python predicates. The reasoning goes that data scientists are well versed with the language, and would prefer programming in Python to programming in some obscure dialect, or learning a completely new tool.

## PMML perspective

The PMML specification provides two model elements for representing business rules models.
The [`TreeModel`](https://dmg.org/pmml/v4-4-1/TreeModel.html#xsdElement_TreeModel) element can be used if the decision path is well determined (ie. unique), which means that a data record can and shall be matched by a single business rule only.
The [`RuleSetModel`](https://dmg.org/pmml/v4-4-1/RuleSet.html#xsdElement_RuleSetModel) element can be used to address any conceivable decisioning strategy. For example, evaluating a data record against a collection of business rules, and computing the decision by applying some aggregation function (eg. a weighted sum) over the matched business rules.

A predicate is simply a boolean expression. The PMML specification provides three categories of predicate elements:

* Constant predicates. The [`True`](https://dmg.org/pmml/v4-4-1/TreeModel.html#xsdElement_True) element and the [`False`](https://dmg.org/pmml/v4-4-1/TreeModel.html#xsdElement_False) element.
* Primary predicates (`<Field> <Operator> <Value>`). The [`SimplePredicate`](https://dmg.org/pmml/v4-4-1/TreeModel.html#xsdElement_SimplePredicate) element implements comparison expressions (`equal`, `notEqual`, `lessThan`, `lessOrEqual`, `greaterOrEqual` and `greaterThan` operators) and missingness checks (`isMissing` and `isNotMissing` operators). The [`SimpleSetPredicate`](https://dmg.org/pmml/v4-4-1/TreeModel.html#xsdElement_SimpleSetPredicate) element implements set membership expressions (`isIn` and `isNotIn` operators).
* Secondary predicate (`<Predicate> <Operator> <Predicate>`). The [`CompoundPredicate`](https://dmg.org/pmml/v4-4-1/TreeModel.html#xsdElement_CompoundPredicate) element implements boolean logic expressions (`and`, `or`, `xor` and `surrogate` operators).

The expressive power of primary predicates seems rather limiting at first glance. For example, they require the right-hand side of the expression to be a value literal (rather than another field reference or expression/predicate), which rules out direct comparisons between fields (`<Field> <Operator> <Field>`). The workaround is to extract such complex logic into a standalone `DerivedField` element, and re-express the primary predicate in terms of this derived field and its possible value range. For example, a comparison expression `x1 < x2` can be re-expressed as `(x1 - x2) < 0` or `(x1 / x2) < 1`.

## `RuleSetClassifier` model type

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package version 0.38.0 introduced the [`sklearn2pmml.ruleset.RuleSetClassifier`](https://github.com/jpmml/sklearn2pmml/blob/master/sklearn2pmml/ruleset/__init__.py) class, which allows data scientists to implement a business rules model as a regular Scikit-Learn classifier.

The complete set of user-specified business rules is presented as an iterable of tuples. The first element is a Python predicate, and the second element is the associated (ie. to be predicted) class label. It is likely that future `sklearn2pmml` package versions add support for more elements, such as the associated class probability distribution.

The search for a matching business rule takes place using the simplest "first hit" strategy. If there is no match, then the default class label (could be `None` to indicate a missing result) is returned instead.

The Python side evaluates business rules using Python's built-in [`eval(expr)`](https://docs.python.org/3/library/functions.html#eval) function. The data record is presented as `X` row array (ie. the shape is `(1, )`). Depending on the backing Scikit-Learn workflow, the cells of this row array may be referentiable by name and/or by position. Wherever technically feasible, name-based cell references should be preferred over positional ones. They are easier to read and write, and do not break if the workflow is reorganized.

Python predicates involving continuous features:

* `X['Petal_Length'] is not None`
* `(X['Petal_Length'] > 0) and (X['Petal_Length'] < 3)`

Python predicates involving categorical features:

* `X['Species'] is None`
* `X['Species'] != 'setosa'`
* `X['Species'] in ['versicolor', 'virginica']`

If the `RuleSetClassifier` model will be used only in Scikit-Learn environment, then Python predicates may take advantage of full language and library/platform functionality. However, if the `RuleSetClassifier` model needs to be converted to the PMML representation, then some limitations and restrictions apply. A great deal of them are temporary, and will be lifted as the Python-to-PMML [expression](https://github.com/jpmml/jpmml-python/blob/master/pmml-python/src/main/javacc/expression.jj) and [predicate translation components](https://github.com/jpmml/jpmml-python/blob/master/pmml-python/src/main/javacc/predicate.jj) of the JPMML-SkLearn library evolve.

## Example workflow

The "iris" dataset is loaded using the [`sklearn.datasets.load_iris()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) utility function. However, the default representation of Scikit-Learn datasets (instance of `sklearn.utils.Bunch`) is too cumbersome for actual work, and needs to be re-packaged.

Feature data (the `Bunch.data` attribute) is renamed (eg. from "sepal length (cm)" to "Sepal.Length") and converted from a 2-D Numpy array to a `pandas.DataFrame` object in order to make name-based cell referencing possible.

The business rules model type can be regarded as a pseudo-unsupervised learning method, because user-specified class labels are completely ignored during training.
This is demonstrated by skipping target data (the `Bunch.target` attribute), and constructing a `pandas.Series` object that is completely filled with `None` values (as opposed to "setosa", "versicolor" or "virginica" string values).
The `RuleSetClassifier.fit(X, y)` method would be perfecty fine to accept `y = None`. The one and only benefit of supplying `y = Series(..)` instead is about customizing the name of the target field in the resulting PMML document.

``` python
from pandas import DataFrame, Series
from sklearn.datasets import load_iris

import numpy

iris = load_iris()

iris_X = DataFrame(iris.data, columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"])
iris_y = Series(numpy.empty(shape = (iris_X.shape[0], ), dtype = object), name = "Species")
```

The implementation of the aforementioned decision tree as a business rules model:

``` python
from sklearn2pmml.ruleset import RuleSetClassifier

classifier = RuleSetClassifier([
  ("X['Petal_Length'] < 2.45", "setosa"),
  ("X['Petal_Width'] < 1.75", "versicolor"),
], default_score = "virginica")
```

Converting this model to a PMML document:

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
  ("classifier", classifier)
])
pipeline.fit(iris_X, iris_y)

sklearn2pmml(pipeline, "RuleSetIris-simple.pmml")
```

The PMML representation of the "core" of this model as a [`RuleSet`](https://dmg.org/pmml/v4-4-1/RuleSet.html#xsdElement_RuleSet) element:

``` xml
<RuleSet defaultScore="virginica" defaultConfidence="1.0">
  <RuleSelectionMethod criterion="firstHit"/>
  <SimpleRule score="setosa">
    <SimplePredicate field="Petal_Length" operator="lessThan" value="2.45"/>
  </SimpleRule>
  <SimpleRule score="versicolor">
    <SimplePredicate field="Petal_Width" operator="lessThan" value="1.75"/>
  </SimpleRule>
</RuleSet>
```

Real-life workflows tend to include data pre-processing steps that revert the `DataFrame` object back to a Numpy array, thereby enforcing positional cell references.
For example, the following `sklearn_pandas.DataFrameMapper` step transforms the "named" four-fimensional feature space into an "anonymized" two-dimensional feature space (average flower dimensions), which must be addressed using `X[<column>]`-style field references:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import Alias, ContinuousDomain
from sklearn2pmml.preprocessing import ExpressionTransformer

pipeline = PMMLPipeline([
  ("mapper", DataFrameMapper([
    (["Sepal_Length", "Petal_Length"], [ContinuousDomain(), Alias(ExpressionTransformer("(X[0] + X[1]) / 2"), "avg(Sepal.Length, Petal.Length)")]),
    (["Sepal_Width", "Petal_Width"], [ContinuousDomain(), Alias(ExpressionTransformer("(X[0] + X[1]) / 2"), "avg(Sepal.Width, Petal.Width)")])
  ])),
  ("classifier", RuleSetClassifier([
    ("X[0] < 3.875", "setosa"),
    ("X[1] < 2.275", "versicolor")
  ], default_score = "virginica"))
])
pipeline.fit(iris_X, iris_y)

sklearn2pmml(pipeline, "RuleSetIris-complex.pmml")
```

The averages are calculated using the `sklearn2pmml.preprocessing.ExpressionTransformer` transformer.

By default, the expression string (ie. `(X[0] + X[1]) / 2`) becomes the name of the corresponding derived field. Data scientists can rename derived fields using the `sklearn2pmml.decoration.Alias` decorator as they see fit.
However, in this example, the renaming is required, because the two derived fields use identical expression strings.

Just for the record, expression strings can also be made unique by reordering terms (`X[0] + X[1]` vs. `X[1] + X[0]`) or surrounding them with redundant parentheses (`X[0] + X[1]` vs. `(X[0]) + (X[1])`).
