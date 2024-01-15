---
layout: post
title: "Upgrading Scikit-Learn decision tree models"
author: vruusmann
keywords: scikit-learn
---

## Overview ##

[Decision trees](https://scikit-learn.org/stable/modules/tree.html) and [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html) are some of the most popular model types for ML applications.
Their strengths include simplicity, versatility and interpretability, and getting to know their ins and outs is a prerequisite to a successful career in data science.

Decision trees are suited for solving both classification- and regression-type problems.
The classical example is multinomial classification using the ["iris" dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set):

``` python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y = True)

classifier = DecisionTreeClassifier()
classifier.fit(X, y)
```

This Python code snippet executes fine with all [Scikit-Learn](https://scikit-learn.org) versions published within the last 10 years.

A typical Scikit-Learn model lifecycle:

``` python
# Development environment, using a then-current Scikit-Learn version
_pkl_dump(classifier, "classifier.pkl")

#
# The classifier.pkl file is carried across space and time
# from the development environment to many production environments
#

# Production environment, using the now-current Scikit-Learn version
classifier = _pkl_load("classifier.pkl")

X, _ = load_iris(return_X_y = True)

yt = classifier.predict(X)
```

In practice, a model is trained once, and is used for prediction many times over the years.

Until very recently, the above model serialization and deserialization workflow was extremely reliable.
For example, one could restore a Scikit-Learn 0.17 model trained 10 years ago, and use it for predictions in Scikit-Learn 1.2.2.
However, this is no longer possible with Scikit-Learn 1.3.0 and newer.

A "model breakdown" following a routine software upgrade is definitely a very unpleasant surprise.
For major enterprises, the cost of losing access to their business-critical intellectual property can quickly amount to millions of US dollars.

Scikit-Learn developers see this "model breakdown" as inevitable and provide no straightforward solutions, instead suggesting that data scientists should retrain their models from scratch using the latest library version.
Unfortunately, this is not always feasible, especially with more complex models where the original training dataset and documentation has been lost.

In such a situation, the least painful option is often to remain infinitely stuck with one particular legacy Scikit-Learn version.

## Problem description ##

Any attempt to load a legacy decision tree model shall fail with the following value error:

```
Traceback (most recent call last):
  File "sklearn/tree/_tree.pyx", line 728, in sklearn.tree._tree.Tree.__setstate__
  File "sklearn/tree/_tree.pyx", line 1434, in sklearn.tree._tree._check_node_ndarray
ValueError: node array from the pickle has an incompatible dtype:
- expected: {'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'], 'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'], 'offsets': [0, 8, 16, 24, 32, 40, 48, 56], 'itemsize': 64}
- got     : [('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'), ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')]
```

This is a sanity check by the `Tree.__setstate__()` method to ensure that the unpickled data is structurally complete and valid decision tree data.

Specifically, the `Tree` class requires the value of its `nodes` attribute to be a 9-dimensional array (with the dimensions being `left_child`, `right_child`, .., `weighted_n_node_samples` and `missing_to_to_left`).
Here, however, it finds an 8-dimensional array, where the ninth `missing_go_to_left` dimension is not present.

The `missing_go_to_left` dimension is a Scikit-Learn 1.3 addition to enhance [the decision tree algorithm with basic missing values support](https://scikit-learn.org/stable/modules/tree.html#missing-values-support).

The Python language and runtime provide relatively simple means to modify class definitions on-the-fly.
So, would it be possible to satisfy this new structural requirement using reflection tools?

Unfortunately, the answer is "no", due to two overwhelming technical complexities:
1. The `Tree` class is a [Cython](https://cython.org/) class, rather than an ordinary Python class. Cython class definitions are effectively impervious to outside manipulation, especially using tools and techniques that are available to ordinary end users.
2. The `Tree.nodes` attribute holds a single n-dimensional [Numpy structured array](https://numpy.org/doc/stable/user/basics.rec.html), rather than an n-element collection of Numpy ndarrays. A Numpy structured array allows updating the values of each dimension, but not adding or removing the definitions of dimensions themselves.

To make things worse, the attributes of Cython classes are not directly accessible via common reflection methods such as `hasattr`, `getattr` or `setattr`:

``` python
tree = classifier.tree_

# Do not exist here
assert not hasattr(tree, "nodes")
assert not hasattr(tree, "values")

state = tree.__getstate__()

# Exist here
assert "nodes" in state.keys()
assert "values" in state.keys()
```

It appears to be the case that the state of a `Tree` object can only be reliably read and written using `Tree.__getstate__()` and `Tree.__setstate__(state)` methods, respectively.

## Workaround idea and implementation ##

The suggested workaround is to perform the manual modification of the problematic Numpy structured array outside of the pickle protocol layer.

The input is a legacy decision tree model in a pickle file.

Two Scikit-Learn environments will be used.
The first is Scikit-Learn <= 1.2.2 that can unpickle this file without any problems, and the second is Scikit-Learn >= 1.3.0 that cannot do so.

The goal is to turn the input pickle file into a completely new pickle file that could be unpickled in Scikit-Learn >= 1.3.0.

The workflow has tree steps:

1. Loading the pickle file in the legacy environment and splitting it into two parts - a **version-<u>in</u>dependent** Python part (ie. the `DecisionTreeClassifier` class) and a **version-dependent** Cython part (ie. the `Tree` class).
2. Manually adjusting the state of the Cython object. When Scikit-Learn framework-level security and sanity checks are not active, then it becomes possible to perform arbitrary data structure molding. The changed state is saved as a new Cython object.
3. Loading the two parts from pickle files in the new environment and recombining them back into a fully-functional decision tree object.

### Partitioning a functional decison tree object into "DT-shell" and `Tree` objects

``` python
classifier = _pkl_load("classifier.pkl")

tree = classifier.tree_

# Drop the tree attribute
delattr(classifier, "tree_")

_pkl_dump(classifier, "classifier_shell.pkl")

# Print Python state
print("Tree(n_features = {}, n_classes = numpy.asarray({}), n_outputs = {})".format(tree.n_features, tree.n_classes, tree.n_outputs))

# Extract and dump Cython state
tree_state = tree.__getstate__()

_pkl_dump(tree_state, "tree_state.pkl")
```

The value of the `tree_` attribute is extracted, after which the attribute is deleted.

There is no point in trying to dump the `Tree` object in pickle data format, because it is a sealed "black box"-type object, whose unpickling behaviour in Scikit-Learn >= 1.3.0 is to raise the above `ValueError`.

It follows that the deconstruction of the `Tree` object should be continued until the problematic `dtype` sanity check gets excluded from executable code paths.

Luckily enough, such a stable state is easily retrievable using the `Tree.__getstate__()` method. The returned value is a standard Python `dict` object that contains four items: `max_depth`, `node_count`, `nodes` and `values`.
In order to upgrade/downgrade the tree, its `nodes` item must be brought up-to-date with the target Scikit-Learn version. The other three items will remain unchanged.

Since the original `Tree` object was left behind, a new one will have to be constructed in the new place and time, by invoking the `Tree` constructor.

Quick inspection of its source code reveals three constructor parameters: `n_features`, `n_classes` and `n_outputs`.
Surprisingly enough, none of them is present in the stable state that is returned by the `Tree.__getstate__()` method, which means that they must be carried over using alternative means.
In this exercise, they are simply sent to the console print-out.

This step results in `classifier_shell.pkl` and `tree_state.pkl` pickle files that can be unpickled using an arbitrary Scikit-Learn version (in an arbitrary Python environment).

<details markdown=block>
<summary markdown=span>Why do Scikit-Learn pickle files get outdated, and how to fix them?</summary>
<br>
If the original `DecisionTreeClassifier` object was trained using a very old Scikit-Learn version, then it is possible that unpickling may raise a `ModuleNotFoundError` stating `No module named 'sklearn.tree.tree'`.

This error is caused by the evergoing (re)organization of the Scikit-Learn module structure.
More specifically, module hierarchies have been systematically flattened over the years - all public classes have been brought from the third or even fourth level to the second level.
For example, the `sklearn.tree.tree.DecisionTreeClassifier` class has been truncated to `sklearn.tree.DecisionTreeClassifier`.

Such import errors can be resolved by re-mapping the old long/problematic module name to its new short/non-problematic form:

``` python
import importlib
import sys

legacy_modulename = "sklearn.tree.tree"
modulename = "sklearn.tree"

# Define a run-time alias
sys.modules[legacy_modulename] = importlib.import_module(modulename)
```
</details>

### Extending the `Tree` object with a `missing_go_to_left` field

``` python
import numpy

tree_state = _pkl_load("tree_state.pkl")

nodes = tree_state["nodes"]

shape = nodes.shape
dtype = nodes.dtype

# Create a Scikit-Learn >= 1.3.0 compatible data type
new_dtype = numpy.dtype(dtype.descr + [('missing_go_to_left', '|u1')])

new_nodes = numpy.empty(shape, dtype = new_dtype)

# Copy existing dimensions
for field_name, field_dtype in dtype.fields.items():
  new_nodes[field_name] = nodes[field_name]

# Append a new dimension
new_nodes["missing_go_to_left"] = numpy.zeros(shape[0], dtype = numpy.uint8)

tree_state["nodes"] = new_nodes

_pkl_dump(tree_state, "tree_state-upgraded.pkl")
```

The `tree_state` object is essentially a `dict` of two Numpy arrays - the `nodes` item holds a Numpy structured array, whereas the `values` item holds a plain Numpy ndarray. 
The `tree_state` object therefore only has Numpy library dependency.
It does not have any Scikit-Learn library dependency, and can therefore be loaded in any Python environment, with or without a Scikit-Learn library available.

A recipe for upgrading Numpy structured arrays is given in this excellent StackOverflow answer: [https://stackoverflow.com/a/25429497](https://stackoverflow.com/a/25429497)

In brief, the first operation is to define a new `dtype` object by appending a new dimension to an existing 8-dimensional `dtype` object.
This extra dimension is called `missing_go_to_left`and its canonical element type is `bool`, which is conventionally mapped to an unsigned 1-byte integer.

Once the new `dtype` has been defined, a new Numpy structured array can be allocated based on it, and the content of the old array can be copied field by field.

The final operation is to fill the new `missing_go_to_left` field.
According to the specification, the fill values should be `0` or `1`, indicating whether the missing values should be sent to the right branch or the left branch, respectively, at any given split point.

The above Python code snippet fills all slots with the `0` value.
This is justifiable, because the intended applicability domain of this model is restricted to dense datasets. In other words, the training dataset did not contain any missing values, and hence they should not be present in any testing datasets either.
Given that absence of missing values there is no significance to the fill values of the `missing_go_to_left` field, because the respective business logic will never get triggered anyway.

If the intended applicability domain includes sparse datasets, then realistic fill values must be used.
They can be reconstructed after the fact, because they reflect the "majority's way" at each split point (ie. "samples with a missing value should be sent to the left branch if the left branch saw more training dataset samples than the right branch, and to the right branch otherwise").
The `nodes` array provides all the necessary information for this evaluation in its `left_child`, `right_child` and `n_node_samples` fields.

Finally, after creating and populating the new Numpy structured array, it is used to upgrade the `tree_state` object.
The latter is dumped into a `tree_state-upgraded.pkl` file (the "upgraded" filename suffix indicates Scikit-Learn >= 1.3.X compatibility).

### Recombining "DT-shell" and `Tree` objects back into a functional decision tree object

``` python
from sklearn.tree._tree import Tree

import numpy

classifier = _pkl_load("classifier_shell.pkl")

tree_state = _pkl_load("tree_state-upgraded.pkl")

tree = Tree(n_features = 4, n_classes = numpy.asarray([3]), n_outputs = 1)
tree.__setstate__(tree_state)

# Re-assign the tree attribute
classifier.tree_ = tree

_pkl_dump(classifier, "classifier-upgraded.pkl")
```

This step requires a Python environment where a Scikit-Learn >= 1.3.0 library is available.

The low-level unpickler component attempts to resolve and load the fully-qualified class name `sklearn.tree._tree.Tree` into some kind of Python class definition.
When doing so, it must first come across the upgraded Cython class that supports a 9-dimensional `nodes` attribute.

There is not much room for error in this regard anyway.
If the unpickler component first comes across the legacy Cython class, then unpickling shall soon raise a value error that is very similar to the original one.
The main difference is in the error message, which now complains about the presence of an unnecessary/unsupported `missing_go_to_left` field (whereas originally it complained about its absence).

The `Tree` constructor requires arguments for `n_features`, `n_classes` and `n_outputs` parameters.
In general, these can be deduced from the dataset description (metadata).
For example, the "iris" classification data table has four columns (ie. `n_features = X.shape[1]`) and it is a single-output (ie. `n_outputs = 1`) multinomial classification type target, where the number of classes is three (ie. `n_classes = 3`).
The only gotcha is about the formatting the `n_outputs` value, which is expected to be a single-element Numpy array, not a Numpy scalar.

Alternatively, if the nature of the dataset is not so well known (eg. a historical dataset), the values of the three parameters can be extracted from the original `Tree` object using reflection tools.

After the `Tree` object is successfully created, it is assigned to the `DecisionTreeClassifier.tree_` attribute.
With this, the decision tree model has been promoted from its non-functional "DT-shell" state back to a fully-functional state, and it is dumped into the `classifier-upgraded.pkl` file.

### Verification

``` python
from sklearn.datasets import load_iris

import sys

# Use the first command-line argument as the name of the pickle file
classifier = _pkl_load(sys.argv[1])

X, _ = load_iris(return_X_y = True)

y_proba = classifier.predict_proba(X)
print(y_proba)
```

The most straightforward and authoritative way of verifying the upgrade is to use models for prediction.
The legacy model and the upgraded model should yield identical results when inputted with the same testing dataset(s).
Here, the word "identical" should be construed as full numerical equivalence - all approximately 14 to 16 significant digits must match.

It should be noted that the predictions need to be made separately.
For example, first using the `classifier.pkl` file in Scikit-Learn <= 1.2.2, and afterwards using the `classifier-upgraded.pkl` file in Scikit-Learn >= 1.3.0.
Any attempts to load these two pickle files into one and the same Python environment are guaranteed to fail.

## Conclusions ##

This blog post demonstrates how to upgrade a legacy decision tree model to the latest version.
For all practical intents and purposes, the upgraded model is indistinguishable from any newly trained model.

By reversing the process (ie. by deleting the `missing_go_to_left` field from the `Tree.nodes` attribute), it is possible to downgrade a latest decision tree model to the legacy representation.
This is rarely needed, but it is good to know that the option is there.

After minor tweaking, the same upgrade/downgrade workflow will also be applicable to decision tree ensemble models (eg. `RandomForestClassifier` and `GradientBoostingClassifier` classes).
The idea is to extract and modify a list of `tree_state` objects instead of a single `tree_state` object.
