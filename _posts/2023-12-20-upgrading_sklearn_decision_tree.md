---
layout: post
title: "Upgrading Scikit-Learn decision tree models"
author: vruusmann
keywords: scikit-learn
---

## Overview ##

[Decision trees](https://scikit-learn.org/stable/modules/tree.html) and [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html) are some of the most widely used types of models for machine learning (ML) applications.
Their strengths include high adaptability, simplicity and interpretability.

Decision trees are suited for solving both classification- and regression-type problems. This is probably one of the first model types applied by any beginner data analyst.

A universally familiar example is multinomial classification based on the ["iris" dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set):

``` python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y = True)

classifier = DecisionTreeClassifier()
classifier.fit(X, y)
```

This snippet executes with all versions of [Scikit-Learn](https://scikit-learn.org) published within the last 10 years.

In practice, however, a model is trained once with a then-current Scikit-Learn version, and is then archived.
The model is then utilized in various ML applications over the years, by restoring the archived model using the now-current Scikit-Learn version, and applying it to new test data.

A typical Scikit-Learn model lifecycle:

``` python
# Development environment, using a then-current Scikit-Learn version
_pkl_dump(classifier, "classifier.pkl")

# The classifier.pkl file is carried across space and time from the development environment to many production environments

X, y = load_iris(return_X_y = True)

# Production environment, using the now-current Scikit-Learn version
classifier = _pkl_load("classifier.pkl")
yt = classifier.predict(X)
```

Until only recently, the above model serialization & deserialization workflow was extremely reliable.
For example, one could restore an Scikit-Learn 0.17 model trained 10 years ago, and use it for predictions in Scikit-Learn 1.2.2.
However, this is no longer possible with Scikit-Learn 1.3.0 and up.

This "model breaking" occurring between Scikit-Learn versions 1.2.2 and 1.3.0 can be a really unpleasant surprise for organizations who are effectively locked out of their business-critical intellectual property.

Scikit-Learn developers see this "model breaking" as inevitable and provide no straightforward solutions, instead suggesting that users retrain their models from scratch using the latest library.
Regrettably, this is not always feasible, especially with more complex models where the original training data and documentation has been lost.

Often the only practical solution is for organisations to remain infinitely stuck with Scikit-Learn <= 1.2.2, with no realistic way to upgrade to newer versions.

## Problem description ##

Any attempt to load a legacy decision tree model will raise the following ValueError:

```
Traceback (most recent call last):
  File "sklearn/tree/_tree.pyx", line 728, in sklearn.tree._tree.Tree.__setstate__
  File "sklearn/tree/_tree.pyx", line 1434, in sklearn.tree._tree._check_node_ndarray
ValueError: node array from the pickle has an incompatible dtype:
- expected: {'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'], 'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'], 'offsets': [0, 8, 16, 24, 32, 40, 48, 56], 'itemsize': 64}
- got     : [('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'), ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')]
```

This is a sanity check by the `Tree.__setstate__()` method to verify the structure of data loaded from the pickle file.
Specifically, the `Tree` class requires the value of the `nodes` attribute to be a 9-dimensional array (with the dimensions being `left_child`, `right_child`, .., `weighted_n_node_samples` and `missing_to_to_left`).
Here, however, it finds an 8-dimensional array, where the ninth `missing_go_to_left` dimension is absent.

The `missing_go_to_left` dimension is a Scikit-Learn 1.3 addition implemented to enhance [the decision tree algorithm with basic missing values support](https://scikit-learn.org/stable/modules/tree.html#missing-values-support).

The Python language and runtime provide relatively simple means to modify class definitions on-the-fly.
So could the model not be forced to load using reflection tools?

Unfortunately not, due to two additional complexities:
1. The `Tree` class is a [Cython](https://cython.org/) class, rather than an ordinary Python class. This makes its behaviour a lot less susceptible to intervention. It is difficult to modify using common Python script with techniques accessible to normal users.
2. The `nodes` attribute holds a single n-dimensional [Numpy structured array](https://numpy.org/doc/stable/user/basics.rec.html), rather than an n-element collection of Numpy ndarrays. A Numpy structured array allows updating the values of each dimension, but not adding or removing the definitions of dimensions themselves.

## Workaround idea ##

This workaround relies on manual modification of the Numpy structured array outside the pickle protocol layer.

The easiest and most robust way is to divide the Scikit-Learn decision tree into two components.
First, the high-level Python `DecisionTreeClassifier` or `DecisionTreeRegressor` class, the precise state and behaviour of which are directly determined by the Scikit-Learn version in use.
Second, the enclosed Cython `Tree` class with a state comprised of five or so attributes, including the problematic `nodes` attribute.

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

The state of a `Tree` object can only be reliably read and written using `Tree.__getstate__()` and `Tree.__setstate__(state)` methods, respectively.

## Step-by-step implementation ##

This workflow begins from the existing decision tree mudel in pickle data format, generated using the legacy Scikit-Learn version.
We have two Scikit-Learn environments available: any Scikit-Learn <= 1.2.2 capable of deserializing the file without issues, and an Scikit-Learn >= 1.3.0 unable to do so.
The goal is to use the existing pickle file to generate a completely new pickle file which deserializes without problems in Scikit-Learn >= 1.3.0 (ja, conversely, does not deserialize in Scikit-Learn <= 1.2.2).

It's a three-step workflow.
The first step is to load the pickle file in the old environment and divide it into two parts - a version-**independent** Python part (eg. `DecisionTreeClassifier` class) and a version-**dependent** CPython part (eg. `Tree` class).
The second step is to manually adjust the state of the CPython object. Because Scikit-Learn framework-level security and sanity checks are not enabled, arbitrary data structure moldings are available. The adjusted state is saved as a new CPython object.
The third step is to load the two parts from pickle files in the new environment and recombine them into a functional decision tree object.

### Partitioning a functional decison tree object into "DT-shell" and Tree objects

``` python
classifier = _pkl_load("classifier.pkl")

tree = classifier.tree_

# Drop the tree attribute
delattr(classifier, "tree_")

_pkl_dump(classifier, "classifier_shell.pkl")

# Print Python state
print("Tree(n_features = {}, n_classes = numpy.asarray({}), n_outputs = {})".format(tree.n_features, tree.n_classes, tree.n_outputs))

# Extract and dump CPython state
tree_state = tree.__getstate__()

_pkl_dump(tree_state, "tree_state.pkl")
```

The value of the `tree_` attribute is extracted, after which the attribute is deleted.

There is no sense saving the resulting `Tree` object in full in pickle data format, as it is a sealed box type object, deserialization of which in Scikit-Learn >= 1.3.0 will raise the above ValueError.

Therefore, we should continue deconstructing the `Tree` object until the problematic `dtype` sanity check is excluded from executable code paths.
Luckily this stable state is easily achievable using the `Tree.__getstate__()` method. In this particular case it will return a standard Python `dict` object including four entries for the `max_depth`, `node_count`, `nodes` and `values` items.
In order to upgrade the tree, the `nodes` item must be updated. The other three items will remain unchanged, at least in the context of this task.

Because the original `Tree` object was left behind, it will have to be constructed from scratch in the new place and time, by invoking the `Tree` constructor.
By inspecting its source code, we see that the constructor requires three arguments: `n_features`, `n_classes` and `n_outputs`.
Because these are (surprisingly so!) not included in the standardard `__getstate__()` result, we need an alternative way to retain these.
In this exercise we will take the easy way and simply write them down separately.

This step results in two new pickle files which we should be able to load using any version of Scikit-Learn (in any Python environment).

A general note on dealing with outdated pickle files.
If the original `DecisionTreeClassifier` class was trained in a very old version of Scikit-Learn, deserialization may raise a ModuleNotFoundError stating: `No module named 'sklearn.tree.tree'`.
This error message is due to the gradual (re)organization of the Scikit-Learn framework module structure over the years.
Specifically, this has included a flattening of module hierarchies - all third or lower level public classes have been raised to exactly the second level. For example, the `sklearn.tree.tree.DecisionTreeClassifier` has become `sklearn.tree.DecisionTreeClassifier`.

To resolve this ModuleNotFoundError, all we need to do is remap the old long/problematic module name to its new short/non-problematic form:

``` python
import importlib
import sys

legacy_modulename = "sklearn.tree.tree"
modulename = "sklearn.tree"

# Define a run-time alias
sys.modules[legacy_modulename] = importlib.import_module(modulename)
```

### Extending the `Tree` object with a "missing_go_to_left" field

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

The recipe for upgrading Numpy structured arrays is given in this excellent StackOverflow answer: https://stackoverflow.com/a/25429497

In brief, the first step is to define a new `dtype` object by appending a new dimension to an existing 8-dimensional `dtype` object.
This extra dimension is called `missing_go_to_left` with the canonic element type `bool`, to be mapped by convention to an unsigned 1-byte integer.

Once we have the new `dtype`, it can be used to allocate a new Numpy structured array, copying the entire old array over field by field.

The final step is to fill the new `missing_go_to_left` field.
Based on the specification, the value should be `0` or `1`, indicating whether the missing values should be passed on to the left or right tree branch, respectively, at that split point.

Our approach of applying a uniform value of `0` to all elements can also be considered conceptually correct.
After all, the training dataset used in the model had no missing values. Accordingly, if the model is going to be used in its intended application domain, these testing datasets should not include any missing values.
Since there are no missing values, the decision to use `0` or `1` values is immaterial as the respective business logic will never be triggered anyway.

Looking at the Scikit-Learn 1.3.X source code, the decision `missing_go_to_left` is made based on whether the training dataset favored the left or right tree branch at that split point. In other words, the missing values will always be sent to the majority's way.
This balance can be reconstructed after the fact, because the `nodes` array includes all necessary information encoded particularly in the `left_child`, `right_child` and `n_node_samples` fields.
However, we leave this exercise for those interested, as it exceeds the scope of this blog post.

Finally, after creating and populating the new Numpy structured array, it is used to upgrade the `tree_state` object.
The latter is saved in a new `tree_state-upgraded.pkl` file (the "upgraded" file name suffix indicates a Scikit-Learn >= 1.3.X compatible object).

### Recombining "DT-shell" and Tree objects back into a functional decision tree object

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

The operation must be conducted in an environment where the Scikit-Learn >= 1.3.0 library is available. This is needed to ensure that the fully-qualified class name `sklearn.tree._tree.Tree` resolves specifically to a CPython class which will recognise and require the `nodes` attribute.

There is not much room for error in this regard anyway.

If we tried to deserialize the "tree_state-upgraded.pkl" pickle file generated in the previous step using Scikit-Learn <= 1.2.2, the operation would fail with a ValueError similar to the one above.
The only difference being the error text, which now complains about an unnecessary/unsupported extra field `missing_go_to_left` (whereas the above error complained about its absence).

When constructing the new `Tree` object, the parameters `n_features`, `n_classes` and `n_outputs`need to be populated.
In general, these can be deduced from the dataset description (metadata).
For example, our "iris" classification data table has four columns (i.e. `n_features = X.shape[1]`) and it's a single-output (i.e. `n_outputs = 1`) multinomial classification type target, where the number of classes is three (i.e. `n_classes = 3`).
There is just one catch, concerning formatting of the `n_outputs` value, because it expect a single-element Numpy array, not a Numpy scalar.

Alternatively, if the nature of the dataset is not well-described (a historical dataset, for example), the values of the three parameters can be extracted from the `Tree` object, as demonstrated in the first step.

After the `Tree` object is successfully created, it is assigned as an attribute of `DecisionTreeClassifier.tree_`.
This concludes the model upgrade, and the result can be serialized in a "classifier-upgraded.pkl" pickle file.

### Verification

``` python
from sklearn.datasets import load_iris

import sys

X, y = load_iris(return_X_y = True)

# Use the first command-line argument as the name of the pickle file
classifier = _pkl_load(sys.argv[1])

y_proba = classifier.predict_proba(X)
print(y_proba)
```

The most straightforward and authoritative validation is to conduct a prediction using the legacy model and upgraded model based on the same input data, and verify that results are identical. Here, "identical" should be construed as full numerical equivalence - identity of all 14 to 16 digits.

Note that the predictions must be conducted independently.
For example, using the "classifier.pkl" in Scikit-Learn <= 1.2.2, and afterwards using the "classifier-upgraded.pkl" file in Scikit-Learn >= 1.3.0.
There is no point trying to load the two files in a single instance of Python, because deserializing one of the two will always fail due to incompatible Scikit-Learn versions.

## Conclusions ##

In this blog post, we demonstrated in detail how to upgrade a legacy decision tree model to one compatible with the latest Scikit-Learn version.
For all intents and purposes, the upgraded model is indistinguishable from a newly trained model (i.e. a casual observer won't be able to tell the difference between the upgraded model and a newly trained model).

Reversing this process (i.e. deleting `missing_go_to_left` from the `Tree.nodes` attribute) would allow us to downgrade decision tree models to the legacy version. While this reverse application is rare, it is good to know that it is technically possible and relatively easy.

Besides standalone decision tree models, the same upgrade/downgrade procedure can also be applied to decision tree ensemble models, such as `RandomForestClassifier`, `GradientBoostingClassifier`, and others.
For this we would need to adjust the above Python code to export and modify a list of `tree_state` objects instead of the single `tree_state` object above.
