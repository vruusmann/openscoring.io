---
layout: post
title: "Measuring the memory consumption of Scikit-Learn models"
author: vruusmann
keywords: scikit-learn sklearn2pmml memory
---

The size of Python objects has two components - the size of the instance state, plus the size of the Python system overhead state.

The size of the instance state (in bytes) can be measured using the `__sizeof__()` method.
This method was introduced in Python 2.6, and should have canonical implementations available for all built-in types by now. Sadly, popular extension packages have been rather slow in adopting it.

The size of the Python system overhead state cannot be measured directly.
It can be calculated by measuring the total size of a Python object using the [`sys.getsizeof(obj)`](https://docs.python.org/3/library/sys.html#sys.getsizeof) function, and then subtracting the size of the instance state from it.
Typically, it is zero for primitive types and 16 bytes for reference types.

### Scikit-Learn estimators do not (yet-) support the `__sizeof__()` method

Can the in-memory size of Scikit-Learn estimator objects be measured using the `__sizeof__()` method or not?

As of Scikit-Learn version 1.1(.3), the answer appears to be "no", because the following code keeps printing a constant value of 24 or 32 bytes (depending on the Python version) for a wide variety of transformer and estimator types:

``` python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(n_samples = 10000, n_features = 10)

estimator = RandomForestRegressor(n_estimators = 31, random_state = 13)
estimator.fit(X, y)

print("Initial state: {} B".format(estimator.__sizeof__()))

estimator.fit(X, y)

print("Final fitted state: {} B".format(estimator.__sizeof__()))
```

The most damning evidence is that the reported size of the `estimator` object does not change after it has been fitted.

### Scikit-Learn estimator instance state

Scikit-Learn estimator types can exist in two instance states:

1. Initial state. Contains learning instructions.
2. Final fitted state. Contains initial learning instruction, a defensive copy of "employed" learning instructions, plus the full specification of the newly-learned function.

The state change happens when the estimator object is fitted by invoking its `fit(X, y)` method.
A fitted estimator object can be re-fitted multiple times. It is rather uncommon to do so, but technically it simply overwrites the previous fitted state with a new one.

According to [Scikit-Learn conventions](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html), end users can distinguish between these two instance states by checking if the estimator object has any attributes that end with a trailing underscore.

With Python being a loosely typed language, it is impossible to write a universal code snippet that could automatically identify and demarcate the instance state of all naturally occuring Python objects.
However, such code snippet can be written for Scikit-Learn estimator types, because they tend to follow a fairly strict and consistent class layout.

Algorithm for collecting actual instance attributes:

1. List all attributes using the [`dir(obj)`](https://docs.python.org/3/library/functions.html#dir) function.
2. Exclude attributes which are not set for the current object. An attribute which is explicitly set to a `None` value survives.
3. Exclude Python system attributes.
4. Exclude callable (ie. function and method) attributes.
5. Exclude property attributes.

Implementation:

``` python
import types

def is_instance_attr(obj, name):
  if not hasattr(obj, name):
    return False
  if name.startswith("__") and name.endswith("__"):
    return False
  v = getattr(obj, name)
  if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType, types.MethodType)):
    return False
  # See https://stackoverflow.com/a/17735709/
  attr_type = getattr(type(obj), name, None)
  if isinstance(attr_type, property):
    return False
  return True

def get_instance_attrs(obj):
  names = dir(obj)
  names = [name for name in names if is_instance_attr(obj, name)]
  return names
```

In this point, there is some hope that the missing `BaseEstimator.__sizeof__()` method can be emulated by iterating over the values of actual instance attributes, and summing their sizes:

``` python
def sklearn_sizeof(obj):
  sum = 0
  names = get_instance_attrs(obj)
  for name in names:
    v = getattr(obj, name)
    v_type = type(v)
    v_sizeof = v.__sizeof__()
    sum += v_sizeof
  return sum

print("Instance state: {} B".format(sklearn_sizeof(estimator)))
```

Unfortunately, closer inspection of `RandomForestRegressor` attributes reveals serious issues and irregularities:

| Attribute | `type(v)` | `v.__sizeof__()` |
|-----------|-----------|------------------|
| `_abc_impl` | `_abc._abc_data` | 48 |
| `_estimator_type` | `str` | 58 |
| `_required_parameters` | `list` | 40 |
| `base_estimator` | `sklearn.tree.DecisionTreeRegressor` | 24 |
| `base_estimator_` | `sklearn.tree.DecisionTreeRegressor` | 24 |
| `bootstrap` | `bool` | 28 |
| `ccp_alpha` | `float` | 24 |
| `class_weight` | `NoneType` | 16 |
| `criterion` | `str` | 62 |
| `estimator_params` | `tuple` | 104 |
| `estimators_` | `list` | 296 |
| `max_depth` | `NoneType` | 16 |
| `max_features` | `float` | 24 |
| `max_leaf_nodes` | `NoneType` | 16 |
| `max_samples` | `NoneType` | 16 |
| `min_impurity_decrease` | `float` | 24 |
| `min_samples_leaf` | `int` | 28 |
| `min_samples_split` | `int` | 28 |
| `min_weight_fraction_leaf` | `float` | 24 |
| `n_estimators` | `int` | 28 |
| `n_features_in_` | `int` | 28 |
| `n_jobs` | `NoneType` | 16 |
| `n_outputs_` | `int` | 28 |
| `oob_score` | `bool` | 24 |
| `random_state` | `int` | 28 |
| `verbose` | `int` | 24 |
| `warm_start` | `bool` | 24 |

First, nested estimator objects (`base_estimator` and `base_estimator_` attributes) still contribute a constant value of 24 or 32 bytes.
Second, built-in iterable types such as `list` and `tuple` appear to mis-represent their in-memory size. The situation is understandable for a list of estimator objects (the `estimators_` attribute) but defies all expectations for a list of Python strings (the `_required_parameters` attribute) or a tuple of Python strings (the `estimator_params` attribute).
Third, non-Python objects (eg. CPython and NumPy types) contribute constant values of around one hundred bytes.

The solution is to define a custom "sizeof" function, and apply it (recursively-) to all objects in the object graph.

``` python
import numpy

def deep_sklearn_sizeof(obj, verbose = True):
  # Primitive type values
  if obj is None:
    return obj.__sizeof__()
  elif isinstance(obj, (int, float, str, bool, numpy.int64, numpy.float32, numpy.float64)):
    return obj.__sizeof__()
  # Iterables
  elif isinstance(obj, list):
    sum = [].__sizeof__() # Empty list
    for v in obj:
      v_sizeof = deep_sklearn_sizeof(v, verbose = False)
      sum += v_sizeof
    return sum
  elif isinstance(obj, tuple):
    sum = ().__sizeof__() # Empty tuple
    for i, v in enumerate(obj):
      v_sizeof = deep_sklearn_sizeof(v, verbose = False)
      sum += v_sizeof
    return sum
  # Numpy ndarrays
  elif isinstance(obj, numpy.ndarray):
    sum = obj.__sizeof__() # Array header
    sum += (obj.size * obj.itemsize) # Array content
    return sum
  # Reference type values
  else:
    clazz = obj.__class__
    qualname = ".".join([clazz.__module__, clazz.__name__])
    # Restrict the circle of competence to Scikit-Learn classes
    if not (qualname.startswith("_abc.") or qualname.startswith("sklearn.")):
      raise ValueError(qualname)
    sum = object().__sizeof__() # Empty object
    names = get_instance_attrs(obj)
    if names:
      if verbose:
        print("| Attribute | `type(v)` | `deep_sklearn_sizeof(v)` |")
        print("|---|---|---|")
      for name in names:
        v = getattr(obj, name)
        v_type = type(v)
        v_sizeof = deep_sklearn_sizeof(v, verbose = False)
        sum += v_sizeof
        if verbose:
          print("| {} | {} | {} |".format(name, v_type, v_sizeof))
    return sum
```

The in-memory size of Numpy array objects is calculated using an approximation, where the number of array slots is multiplied by the size of an idealized array element.
This approximation is totally safe and appropriate, because Scikit-Learn only deals with dense 2-D and 3-D Numpy arrays of primitive values.

The `deep_sklearn_sizeof(obj, verbose)` utility function may look a bit clumsy, but its runtime behaviour is just fine, as estimator objects are rather narrow and shallow (ie. low number of attributes, low nesting depth).

Reified `RandomForestRegressor` attributes:

| Attribute | `type(v)` | `v.__sizeof__()` | `deep_sklearn_sizeof(v)` |
|-----------|-----------|------------------|--------------------------|
| `_abc_impl` | `_abc._abc_data` |  48 | 16 |
| `base_estimator` | `sklearn.tree.DecisionTreeRegressor` | 24 | 413 |
| `base_estimator_` | `sklearn.tree.DecisionTreeRegressor` | 24 | 413 |
| `estimator_params` | `tuple` | 104 | 657 |
| `estimators_` | `list` | 296 | 25126487 |

The in-memory size of the sample random forest object is 1'650 bytes (1.5 kB) in its initial state, and 25'128'606 bytes (25 MB) in its final fitted state.
The `estimators_` attribute alone is responsible for 99.9% of the added "weight" (from 0 to 25'126'487 bytes), leaving the other 26 attributes with less than 0.1% (from 1'650 to 2'119 bytes).

Inside the object graph, there are 2 decision tree objects in the initial state, and 31 in the final fitted state.

The in-memory size of fitted decision tree objects ranges from ~805'000 bytes to ~820'000 bytes.
The variance stems from the fact that the training data was randomly generated. All the trees are close to fully grown, but some branches of some trees have hit early stopping criteria, leading to small variations in node counts.

### Update

The `deep_sklearn_sizeof(obj, verbose)` utility function has been refactored, and made available in the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package version 0.87.1 as the `sklearn2pmml.util.deep_sizeof(obj, with_overhead, verbose)` utility function:

``` python
from sklearn2pmml.util import deep_sizeof

memory_size = deep_sizeof(estimator, with_overhead = True, verbose = True)
print(memory_size)
```

### Resources

* Python script: [`measure.py`]({{ "/resources/2022-11-09/measure.py" | absolute_url }})