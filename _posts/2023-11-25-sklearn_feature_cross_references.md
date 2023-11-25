---
layout: post
title: "Extending Scikit-Learn with feature cross-references"
author: vruusmann
keywords: scikit-learn sklearn2pmml
---

Scikit-Learn provides the [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) meta-estimator for aggregating a number of transformers and a final model into an easily manageable unit.

As the name suggests, member estimators are laid out and executed sequentially.
The output of some step is routed to the input of the subsequent step. All data flows are shielded from outside access.

The "pipeline" approach adapts well to strictly linear workflows.
However, as soon as there are non-linearities involved, its structural and computational complexity explodes.

One of the simplest, yet most versatile, non-linear operations is feature cross-references (aka feature Xrefs).
Their value proposition is about eliminating repeated work. A feature value shall be computed once, and then made referentiable to any interested party anywhere, anytime.

The savings in computation time alone should be significant.
However, the real savings shall result from simplified and optimized data flows, as the need for maintaining multiple identical argument datasets to feature value computation disappears.
For example, compare referencing PCA results versus hauling around PCA inputs and re-running the same PCA computation over and over again.

Feature cross-references can be implemented in two ways.

First, replacing `Pipeline` meta-estimator with a custom meta-estimator, which supports [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAG)-like layout and execution modes.

The DAG approach is virtually unknown in Scikit-Learn.
There are only a few third-party implementations available, such as [`skdag`](https://github.com/scikit-learn-contrib/skdag) and [`pipegraph`](https://github.com/mcasl/PipeGraph) packages.

Second, keeping the `Pipeline` meta-estimator, but re-arranging its data flows using (custom-) meta-transformer steps.

So far, the best candidate for the job would be the [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) meta-transformer, which can carry columns along the transformer-part of a pipeline using dataset splitting and joining operations.
However, its pipeline embedding patterns are highly complicated and fragile.

## API ##

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides the `sklearn2pmml.cross_reference` module, which allows pipeline steps to communicate with each other and/or with the outside world using the "feature Xrefs" approach.

Main components:
* `Memory`. Shared mutable memory space.
* `Memorizer`. Transformer for "exporting" columns into memory.
* `Recaller`. Transformer for "importing" columns from memory.

A sample workflow:

``` python
from sklearn2pmml.cross_reference import Memory, Memorizer, Recaller

import numpy

X = numpy.asarray([["green"], ["red"], ["yellow"]])

memory = Memory()

memorizer = Memorizer(memory, names = ["color"])
memorizer.transform(X)

#print(memory["color"])

recaller = Recaller(memory, names = ["color"], clear_after = True)
Xt = recaller.transform(None)

assert (Xt == X).all()
```

The same, in Scikit-Learn pipeline form:

``` python
from sklearn.pipeline import Pipeline

memory = Memory()

pipeline = Pipeline([
  ("memorizer", Memorizer(memory, names = ["color"])),
  ("recaller", Recaller(memory, names = ["color"], clear_after = True))
])
Xt = pipeline.fit_transform(X)

assert (Xt == X).all()
```

A "feature Xrefs"-enhanced pipeline is API-wise indistinguishable from ordinary pipelines.

All code changes take place in the training Python script.
They are about inserting a few memorizer and recaller steps between existing transformer steps.
There is no need for re-structuring or re-parameterizing anything major.

The pipeline should be balanced in terms of column memorization and recall operations, so that the memory object is returned to its original state when the pipeline execution completes.
A column that is left uncleared can be regarded as a memory leak.

The "feature Xrefs" approach relies on its own column naming mechanism, which is completely independent of Scikit-Learn column/feature naming mechanisms (eg. the [`set_output` API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html)) and their activation statuses. 

The names of memorized columns should be short and concise, similar to Python's variable names.
Their purpose is to inform the data scientist about the business intent rather than [SLEP007](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep007/proposal.html)-style derivation path.
For example, `red_light_on` is preferable to `ohe_light_red`.

### Memory

Memorizer and recaller objects can communicate over any [item assignment](https://docs.python.org/3/reference/datamodel.html#emulating-container-types)-capable object such as dictionaries or Pandas' data frames.
However, when wrapped into a pipeline, several extra functional requirements come into play that can only be satisfied by designing and implementing a special-purpose memory class.

The primary requirement is object identity preservation during copy operations.

Scikit-Learn habitually duplicates estimators using the [`sklearn.base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) utility function, which defaults to "deep copy" semantics (ie. the copy operation is propagated to each object in the object graph).
This behaviour is a deal-breaker for the "feature Xrefs" approach, which requires that all memorizer and recaller objects stay associated with exactly the same memory object that they were parameterized with during construction.

Avoiding or suppressing the use of the `clone` utility function is not possible.
The problem of keeping memory-type attributes constant can be addressed either by teaching estimator classes to defend their memory-type attributes or by teaching the memory class to defend itself.
The latter option seems more appropriate in every way.

The secondary requirement is object state minimization during persistence operations.

The memory object should not reference any objects (whether directly or indirectly) besides the ones that have been explicitly assigned to it.
Moreover, any objects that can be classified as "data" should be completely excluded from the persistent state, because they vary from one pipeline execution to another.

These two requirements can be met by wrapping the elementary memory object into an `Memory` object:

``` python
from sklearn2pmml.cross_reference import Memory

memory = Memory(dict())
```

Both `Memory.__copy__()` and `Memory.__deepcopy__(memo)` methods return a reference to the current instance (ie. `self`), thereby preventing Python's `copy` module from replacing it during "shallow copy" or "deep copy" operations.

``` python
from sklearn.base import clone

# Wrap the elementary memory object
memory = Memory(dict())

memorizer = Memorizer(memory = memory, names = ["color"])

memorizer_clone = clone(memorizer)

# Assert memory reference stayed the same
assert memorizer_clone.memory is memory, "Not the same as memory"
assert memorizer_clone.memory is memorizer.memory, "Not the same as memory.memory"
```

On a side note, debugging object graphs might be tricky, because it requires the use of identity checks (ie. Python's `is` keyword) rather than equality checks (ie. Python's `==` operator) and/or print statements.

There are no restrictions on the overall number of `Memory` objects.
Most applications should be adequately served by a single one. But advanced applications may create and use several in order to serve specific controllability, observability, etc. goals.

The `Memory.__getstate__()` method returns a modified state, where the elementary memory object has been cleared of any data.

``` python
import pickle

# Wrap the elementary memory object
memory = Memory(dict())

memorizer = Memorizer(memory = memory, names = ["color"])
memorizer.fit_transform(X)

# Assert data is present
assert len(memorizer.memory) == 1
assert "color" in memorizer.memory

# Emulate migration between Python environments (eg. from dev to prod)
memorizer = pickle.loads(pickle.dumps(memorizer))

# Assert data has been cleared
assert len(memorizer.memory) == 0
assert "color" not in memorizer.memory
```

Memorized columns are guarded against tampering by proxying all read and write operations through extra copying.

### Memorizer

The `Memorizer` (pseudo-)transformer exports the dataset into named variables in memory.

The biggest API gotcha is that the `Memorizer.transform(X)` method returns an empty Numpy array (to be exact, a Numpy array with dimensions `(n_samples, 0)`).
This is so to enable simpler and shorter pipeline embedding patterns.

The number of memorizable columns is typically small compared to the total number of columns that come into existence during pipeline execution (a few vs. hundreds or thousands).
Furthermore, their lifecycles should be planned individually (when to grab, when to release) in order to minimize memory footprint.

The difficulty relates to pinpointing and extracting the right columns.

The canonical pattern is to split the dataset into two using a `FeatureUnion` meta-transformer.
In one branch, the input dataset is filtered down to the desired composition and memorized.
In the other branch, it will be passed through as-is.

A "passthrough" memorizer step:

``` python
from sklearn.pipeline import FeatureUnion

memorizer_union = FeatureUnion([
  ("memorizer", Memorizer(memory = memory, names = ["color"])), # Yields an empty array
  ("passthrough", "passthrough") # Yields the input array unchanged
])
```

The same, with an additional column filtering sub-step:

``` python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

memorizer_union = FeatureUnion([
  ("selector_memorizer", make_pipeline(
    # Keep the first column, drop all the other columns
    ColumnTransformer([
      ("selector", "passthrough", [0])
    ], remainder = "drop"),
    # Memorize the column
    Memorizer(memory = memory, names = ["color"])
  )),
  ("passthrough", "passthrough")
])
```

The `Memorizer` transformer has two operating modes, which can be toggled using the `Memorizer.transform_only` attribute.
The first and the default mode is to perform memorization only during the "transform" pass. The second mode is to perform memorization twice, during both "fit" and "transform" passes.

### Recaller

The `Recaller` (pseudo-)transformer imports named variables from memory into a new dataset.

The `Recaller.transform(X)` method ignores any arguments passed to it.
The ideal behaviour for this method - to achieve symmetry with its counterpart, the `Memorizer.transform(X)` method - would be to raise an error if the `X` argument is not an empty dataset.
Again, the symmetry is broken for the sake of improved embeddability.

The canonical pattern is to join two (or more) datasets into one using the `FeatureUnion` meta-transformer.
In one branch, the input dataset gets replaced with the recalled dataset.
In the other branch, it will be passed through as-is.

A "passthrough" recaller step:

``` python
from sklearn.pipeline import FeatureUnion

recaller_union = FeatureUnion([
  ("recaller", Recaller(memory = memory, names = ["color"])), # Yields a new array
  ("passthrough", "passthrough") # Yields the input array unchanged
])
```

The ordering of feature union branches is free.
If the columns need to be addressed positionally afterwards, then it makes sense to insert the `Recaller` transformer to the first position.
The dimensions of its sub-dataset are fixed (ie. `(n_samples, len(Recaller.names))`), and it will be straightforward to adjust the column offsets of subsequent sub-datasets accordingly.

A column can be recalled from memory any number of times.
However, the final `Recaller` object using it should have its `Recaller.clear_after` attribute set to `True` in order to remove the now-unnecessary variable(s) from memory.
