---
layout: post
title: "Extending Scikit-Learn with UDF expression transformer"
author: vruusmann
keywords: scikit-learn sklearn2pmml jpmml-python jpmml-sklearn
---

A canonical workflow can be segmented into three stages:
1. Data pre-processing. Transforming data from real-life schema to modeling algorithm schema.
2. Modeling. Establishing the (approximate-) mathematical relationship between features and the label.
3. Prediction post-processing. Transforming prediction from modeling algorithm schema back to real-life schema. 

Everyday workflows skip out on both ends.
Typically, they start close to the modeling stage, and stop right after it.
One reason why it happens is the lack of adequate feature engineering and decision engineering tools.

## Feature engineering vs feature transformation ##

Inside the data pre-processing stage, there are two sub-stages:
1. Feature engineering. Deriving new features based on existing features. Manual, ad hoc activity.
2. Feature transformation. Making features compliant with specific requirements. Automated activity.

Feature engineering always precedes feature transformation.
It acts on real-life data.
For example, the values of a string column are accessible as Python strings, and they can be processed using Python's built-in operators and functions.

Feature transformation maps values from one value space to another value space following some mathematical or statistical procedure.
The main use case is ensuring compliance with modeling algorithm requirements.
For example, all the (numeric-) inputs to a linear model should be standardized. Otherwise, the convergence (towards the solution) will be hampered, and the estimated beta coefficients will be meaningless.

The feature transformation needs of Tabular ML applications can be satisfied using a limited number of algorithms such as scaling, discretization and encoding.
Most ML frameworks provide correct and efficient implementations right out of the box.
There is rarely any reason for coding up something extra from scratch.

The dichotomy between the two sub-stages is highly pronounced in AutoML.

State-of-the-art AutoML tools are incapable of feature engineering, but are highly proficient in feature transformation.
To illustrate, they lack the imagination to generate feature crosses or feature ratios.
However, when such synthetic features are presented to them, they will aptly work out which statistical procedure(s) will amplify the signal further.

## Scikit-Learn perspective ##

Scikit-Learn collects all its data pre-processing tools into the `sklearn.preprocessing` module.
The offering is narrow (at least when compared to the offering of modeling algorithms), but covers all the fundamentals.

As of Scikit-Learn version 1.2(.0), there are 18 transformer classes available.
Two of them can be used for feature engineering purposes. The remaining 16 cannot, because they are pure-blood feature transformers.

First, the [`sklearn.preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) class derives features using the multiplicative operator (`*`).
A feature multiplied by itself yields a power feature.
A feature multiplied by some other feature yields a so-called interaction feature.

The `PolynomialFeatures` transformer generates new features using polynomial combination, which quickly exhausts all computational and memory resources.
In the feature engineering mode, it should be applied to feature pairs or triplets, expressly suppressing the generation of unnecessary terms.

For example, interacting a continuous feature with a categorical feature:

``` python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

transformer = Pipeline([
  # Prepare columns
  ("mapper", ColumnTransformer([
    ("cont", "passthrough", [0]),
    ("cat", OneHotEncoder(), [1])
  ])),
  # Interact prepared columns
  ("interactor", PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False))
])
```

Second, the [`sklearn.preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) class derives features using the user-supplied function.

Its API documentation advertises that the `func` argument can be any callable.
However, in practice, only stateless (aka idempotent) functions will do.
The reason behind this statelessness requirement is that the function transformer does not inform the callable whether it is being called in the `fit(X)` or `transform(X)` method context.

For example, the [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) transformer calculates the mean and variance of the training dataset during fitting, and persists them as `StandardScaler.mean_` and `StandardScaler.var_` attributes, respectively, for future transforms on testing datasets.

At first glance, it appears that it should be possible to replace standard scaler with a function transformer, where the `func` argument is the [`sklearn.preprocessing.scale`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html) utility function.

The results between the two will be in perfect agreement when calling the `fit_transform(X)` method with the complete training dataset.
However, the results will be completely different when calling the `transform(X)` method with individual data samples of the training dataset, or with a testing dataset, because the function transformer forgets the calculated mean and variance values right after returning.

``` python
from sklearn.preprocessing import scale, FunctionTransformer, StandardScaler

import numpy

scaler = StandardScaler()
func_scaler = FunctionTransformer(scale)

X_train = numpy.asarray([[4], [-3], [5]])

# Same results
print(scaler.fit_transform(X_train))
print(func_scaler.fit_transform(X_train))

X_test = numpy.asarray([[1], [-2]])

# Different results
print(scaler.transform(X_test))
print(func_scaler.transform(X_test))
```

## Scikit-Learn function transformer ##

Suppose that there is a stateless **user-defined function** (UDF).
Does it mean that it can be integrated into a Scikit-Learn pipeline by simply wrapping it into a `FunctionTransformer` object?
By default, the answer is negative, because the two cannot be bound together in a persistent way.

Scikit-Learn developers recommend using Python's built-in pickle data format for short-term persistence needs.

The persistent state of Python functions is their (fully qualified-) name.
The persistent state of Python anonymous functions aka lambdas is undefined, and they get rejected.

These claims are easy to verify by dumping function objects using [pickle protocol version `0`](https://docs.python.org/3/library/pickle.html#data-stream-format) (the original "human-readable" protocol):

``` python
import numpy
import pickle

def _udf(X):
  return numpy.exp(X)

udf_str = pickle.dumps(_udf, protocol = 0)
print(udf_str)
```

Indeed, the print-out reads as `b'c__main__\n_udf\np0\n.'`.
There is no trace of the function body, or the actual business logic contained therein.

The pickling operation stores the function name, the inverse unpickling operation loads the name and attempts to resolve it in the current namespace.

The `FunctionTransformer` class does not interfere with this procedure.
It is the application's responsibility to ensure that the resolution succeeds, and yields the intended Python function.

A failed name resolution operation raises an attribute error:

``` python
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(_udf)

# Write into a string
pkl_str = pickle.dumps(transformer, protocol = 0)

# Remove UDF from the current namespace
del _udf

# Read back from the string
# Raises an AttributeError: Can't get attribute '_udf' on <module '__main__'>
transformer = pickle.loads(pkl_str)
```

Making guarantees about name resolution requires library approach.
The `FunctionTransformer` transformer is very suitable for working with stable, third-party library functions such as [Numpy universal functions](https://numpy.org/doc/stable/reference/ufuncs.html), where the only source of error can be a missing import statement.
If the application has custom data pre-processing needs, then it should get started with its own supporting UDFs library.

## Optimal programming model ##

As of Scikit-Learn 1.2(.0), there are no formalized tools or guidelines for packaging supporting UDFs.
In principle, persistence issues could be fixed by subclassing the `FunctionTransformer` class, and overriding its shallow "function name"-based pickling behaviour with a deep "full function source code"-based one.

Unfortunately, the `FunctionTransformer` class suffers from a major conceptual issue that makes it unappealing as a feature engineering platform.
Namely, this transformer (just like any other Scikit-Learn transformer or model) uses a 2-D matrix-oriented programming model, which promotes computational efficiency over flexibility.

Feature engineerings deals with individual data samples.
Therefore, it would be desirable to use a 1-D row-oriented programming model instead.

The upside is improved productivity.
Replacing Numpy functions with plain Python language constructs clarifies business logic, and rules out many categories of Numpy-related programming mistakes.
For example, replacing the [`numpy.where(...)`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) utility function with the [conditional expression](https://peps.python.org/pep-0308/), or replacing [`numpy.logical_and(...)`](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html) and [`numpy.logical_or(...)`](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html#numpy.logical_or), etc. utility functions with boolean expressions. 

The downside is potential performance loss.
Missing out on vectorized math operations is unfortunate.
However, there should be no macroscopic effect to it, because in the grand scheme of things, the computational cost of the data pre-processing stage is low compared to the modeling stage.
A few extra loops cannot shift this balance much.

## SkLearn2PMML expression transformer ##

The `sklearn2pmml` package provides the `sklearn2pmml.preprocessing.ExpressionTransformer` class since its early days.
Starting from the SkLearn2PMML version 0.91, it has gained full UDF support, which makes it a viable replacement for the `FunctionTransformer` class in all Scikit-Learn pipelines.

Main advantages:
1. Fully persistable in pickle data format.
2. Isolated execution environment. Ability to audit third-party UDFs before their use.
3. Simplified programming model (1-D row or direct scalar variables).
4. PMML compatible.

The `expr` argument is the evaluatable Python expression in one of the supported representations (see below).

This expression is evaluated using Python's built-in [`eval(expr)`](https://docs.python.org/3/library/functions.html#eval) function in a custom namespace, which contains `math`, `numpy` and `pandas` module imports, and a sole `X` variable that represents the current data sample.
The `ExpressionTransformer.transform(X)` method creates and manages a separate custom namespace object during each call.

Main call sequence:

``` python
env = dict()

exec("import math", env)
exec("import numpy", env)
exec("import pandas", env)

env["X"] = [-1, -1.5]

result = eval("numpy.sign(X[0]) == numpy.sign(X[1])", env)
print(result)
```

The use of the `eval()` function is considered a major security risk.
This should not be the case here, because these three module imports are assumed to be safe, and there is no way to access other namespaces.

When facing an unknown or untrusted `ExpressionTransformer` object, then it is recommended to print out its `expr` attribute, and take note of any high-risk activity such as importing system modules.

However, the best security guarantee can be obtained fully automatically, by attempting conversion into a Predictive Model Markup Language (PMML) document (see below).
The PMML representation of models and transformers is absolutely safe and secure, because the language is Turing-incomplete, and relies on a small standard library for complex calculations.

A PMML conversion error therefore signals that the expression was either syntactically incorrect or contained some instruction that went beyond the PMML scope.
For example, the conversion fails if the expression references any Pandas' IO-related utility functions such as `pandas.read_clipboard()`, `pandas.read_csv(path)`, `pandas.read_pickle(path)` etc.

### Inline expression

The inline string representation is suitable for simple transformations, where the business logic fits conveniently on a single line.

By convention, the data sample is mapped to the `X` variable.
The syntax for accessing data sample elements depends on the type of the data matrix that was passed to the `ExpressionTransformer.transform(X)` method.
Numpy arrays support only positional indexing, whereas Pandas' data frames support both positional and label-based indexing.

The expression must yield a scalar value.
A missing result can be indicated by returning `numpy.NaN` for numeric types, and `None` for non-numeric types.

For example, checking if two elements have the same [sign](https://en.wikipedia.org/wiki/Sign_function) or not:

``` python
from pandas import DataFrame
from sklearn2pmml.preprocessing import ExpressionTransformer

X = DataFrame([
  [-1, 0],
  [-1, -1.5],
  [1, 2]
], columns = ["a", "b"])

transformer = ExpressionTransformer("numpy.sign(X['a']) == numpy.sign(X['b'])")
Xt = transformer.fit_transform(X)
print(Xt)
```

### Inline UDF

The inline string representation is easy to develop, but not so easy to test and maintain.
For example, the Python interpreter regards it as just another string literal, and does not perform any syntactic or semantic checks on its contents.
If the expression is invalid, then it will typically go unnoticed until the `ExpressionTransformer.transform(X)` method is called for the first time.

The robustness of the Python script can be improved by extracting the inline string expression into an UDF.

This UDF must be formatted as a static top-level function in the current module.
Its signature must declare a sole `X` parameter.

``` python
import inspect
import numpy

def _row_eq_sign(X):
  """Checks if two elements have equal signs.

  Parameters:
  X -- a two-element list or list-like
  """
  return numpy.sign(X['a']) == numpy.sign(X['b'])

transformer = ExpressionTransformer(inspect.getsource(_row_eq_sign))
Xt = transformer.fit_transform(X)
print(Xt)
```

The expression can be refactored into a sequence of statements, and enriched with comments.

UDFs that strive towards PMML compatibility must meet the following constraints:

* Exactly one value statement per block. For example, cannot have two `if` statements one after another (ie. same indentation level), but can have one `if` statement nested inside another `if` statement (ie. different indentation levels).
* All value statement branches must terminate with an explicit `return` statement.
* No loops.
* No raising or catching exceptions.

### Inline expression with supporting UDFs

The inline string and UDF representations work fine with third-party library functions.
However, due to the use of a custom namespace for expression evaluation, they cannot see and call any functions in their immediate vicinity.

The solution exists in the form of the `sklearn2pmml.util.Evaluatable` class, which combines a string expression and its supporting UDFs into a single entity.

This class has `Expression` and `Predicate` subclasses.
All SkLearn2PMML transformer and estimator classes have been updated to accept such expression and predicate objects next to string expressions and predicates, respectively.

The list of supporting UDFs must be collected manually.
The order of elements is not important, and redundant elements are ignored.
It is advisable to keep the list as short as possible, because all UDFs are translated into Python source code and persisted.

Supporting UDFs must be once again formatted as static top-level functions in the current module.
However, they are free to choose any signature they like.

If the goal is to promote reusability, then the sole `X` row-vector parameter should be expanded into a list of scalar parameters, one for each relevant data sample element.
The indexing logic stays put in the inline string expression, because this depends on the pipeline context.

``` python
from sklearn2pmml.util import Expression

def _sign(x):
  if x < 0:
    return -1
  elif x > 0:
    return 1
  else:
    return 0

def _eq_sign(left, right):
  return (_sign(left) == _sign(right))

expr = Expression("_eq_sign(X['a'], X['b'])", function_defs = [_eq_sign, _sign])

transformer = ExpressionTransformer(inspect.getsource(_row_eq_sign))
Xt = transformer.fit_transform(X)
print(Xt)
```

Moving this idea forward, a UDF does not need to be coded up in the current module, as it can be imported from any trusted module or third-party library.
The following assignment trick makes it visible in the current namespace:

``` python
import mylib

_sign = mylib._sign
_eq_sign = mylib._eq_sign

expr = Expression("_eq_sign(X['a'], X['b'])", function_defs = [_eq_sign, _sign])
```

## PMML ##

### Java

The [JPMML-Python](https://github.com/jpmml/jpmml-python) library provides low-level `org.jpmml.python.ExpressionTranslator` and `org.jpmml.python.PredicateTranslator` components for translating Python source code snippets into live `org.dmg.pmml.Expression` and `org.dmg.pmml.Predicate` class model objects, respectively.
The [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) library provides high-level utility functions for commanding them.

For example, the JPMML-SkLearn library can unpickle and convert a `sklearn2pmml.util.Evaluatable` object in a couple lines of Java code:

``` java
import org.dmg.pmml.Expression;
import org.jpmml.model.JAXBUtil;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.Scope;
import org.jpmml.python.Storage;
import org.jpmml.python.StorageUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn2pmml.util.EvaluatableUtil;

SkLearnEncoder encoder = new SkLearnEncoder();

Object pyExpression;

// Read pickle
try(InputStream is = ...){
  Storage storage = StorageUtil.createStorage(is);

  pyExpression = PickleUtil.unpickle(storage);
}

// Define the number and type of data sample elements
Scope scope = defineDataSample(encoder);

Expression pmmlExpression = EvaluatableUtil.translateExpression(pyExpression, scope);

// Write PMML
try(OutputStream os = ...){
  JAXBUtil.marshal(pmmlExpression, new StreamResult(os));
}
```

### Python

The `ExpressionTransformer` class takes inspiration both from Python and PMML worlds.

Specifically, in addition to Python-style `expr` and `dtype` attributes, it supports PMML-style `map_missing_to`, `default_value` and `invalid_value_treatment` attributes for extra controls over expression evaluation.
Their role and effect follows the [`Apply`](https://dmg.org/pmml/v4-4-1/Functions.html#xsdElement_Apply) element specification.

For example, the `map_missing_to` attribute activates a quick pre-check that all inputs (ie. data sample elements) are present, and the `default_value` does the same with the output.
This eliminates boilerplate code, bringing the focus back on the actual business logic.

``` python
from pandas import DataFrame
from sklearn2pmml.preprocessing import ExpressionTransformer 

X = DataFrame([
  ["Alice"],
  ["Bob"],
  ["Carol"],
  [None]
], columns = ["person"])

# Manual missingness checks
transformer = ExpressionTransformer("len(X['person']) if X['person'] is not None else -999")
Xt = transformer.fit_transform(X)
print(Xt)

# Automated missingness checks
transformer = ExpressionTransformer("len(X['person'])", map_missing_to = -999)
Xt = transformer.fit_transform(X)
print(Xt)
```

The `ExpressionTransformer` class does not have API for dumping its contents in the PMML representation.

The workaround is to construct and fit a single-step `sklearn2pmml.pipeline.PMMLPipeline` object, and convert it using the `sklearn2pmml.sklearn2pmml` utility function as usual:

``` python
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

transformer = ExpressionTransformer(...)

pipeline = PMMLPipeline([
  ("transformer", transformer)
])
pipeline.fit(X, None)

sklearn2pmml(pipeline, "Expression.pmml")
```

One may wonder that why does a pipeline that contains a sole stateless transformer need fitting?
Strictly speaking, it does not.
The `PMMLPipeline.fit(X, y)` method is simply used for initializing the `PMMLPipeline.active_fields` attribute that informs the converter about real-life feature names.
If left unset, then `x1`, `x2`, .., `x{m_features}` is assumed.
