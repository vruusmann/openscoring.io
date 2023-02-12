---
layout: post
title: "Benchmarking Scikit-Learn against JPMML-Evaluator"
author: vruusmann
keywords: scikit-learn jpmml-evaluator jpmml-evaluator-python optimization
---

### Overview

A ML framework design is a trade-off between training and deployment efficiency. Finding a balance is hard, because these two application areas are conceptually and technically rather different.

During model training, the data is available in its entirety. The technical challenge is loading this dataset into memory, and running numerical optimization algorithms on it.
Model training does not lend itself easily to non-batch processing approaches, because the effective formulation and testing of statistical hypotheses is not possible without sufficient evidence. However, once available, a model can be updated using mini-batch or stream processing approaches (aka "online learning").

During model deployment, the data is presented over time in variable size chunks. One end of the spectrum is real-time prediction, where the chunk size is one. The other end is batch prediction, where the chunk size could be in millions.
However, there is a practical upper limit to chunk size, as dictated by available computer resources.

Model deployment has room for specialization.

ML frameworks that prioritize deployment are likely to offer additional APIs.
The case in point is real-time prediction, where the technical challenge is evaluating an isolated data record as fast as possible.
Replacing a batch-oriented API with a dedicated data record-oriented API allows the application to avoid unnecessary and expensive interconversions between data records and 1-row data matrices.

##### Scikit-Learn

Scikit-Learn is designed for running classical ML algorithms on relatively small datasets.
The high-level business logic is implemented in Python. All the heavy-lifting is dispatched to the C language layer (Cython and/or C-based libraries such as NumPy, SciPy).

The design is influenced by the desire to maximize the usage of vectorized math operations.
Vectorization shines during model training when the data is presented as a single large data matrix.
However, it loses its luster (to varying degrees) during model deployment, when the data is presented as multiple variable-size data matrices.
Vectorization gains disappear when dealing with 1-row data matrices.

The scaling properties of transformer and model types can be deduced from their "vectorizability".

Vectorized math is only possible with (floating point-) numeric features.
If the dataset contains complex data type features (eg. categoricals, temporals, user-defined data types), then the transformation to numeric representation is likely to involve computations that are not vectorizable.
For example, the binarization of categorical string features using one-hot encoding involves compiling a vocabulary and performing string value lookups against it.

The prediction algorithm of many popular model types is either fully (eg. linear models, neural networks, support vector machines) or to a great degree (eg. ensemble models) expressible in terms of matrix (linar algebra) operations.
There are only a few model types that can do without. The best example are decision trees, which rely on conditional logic instead.

Scikit-Learn holds data using Numpy arrays that support different representations.
The default data representation is dense, which has low computational overhead, but requires large contiguous blocks of memory.
As the size of the dataset increases, this requirements becomes harder to satisfy.

The data and the execution algorithm can be kept "in-core" longer by changing the data representation from dense to sparse.
For maximum effect, must choose a specific sparse matrix implementation depending on the compression axis (rows vs. columns), and the compression rate (high-density vs. low-density).

Switching the data representation from dense to sparse typically introduces very significant computational overhead, because cell values must be retrieved individually.
The extra work is carried out automatically deep inside the C language layer. The application developer perceives a performance degradation, but cannot do much about it.

Some Scikit-Learn algorithms contain different execution paths for dense vs. sparse data. Some may even refuse to deal with the latter.

Another expensive operation that is often overlooked is the concatenation of child transformer results by the parent (meta-)transformer.
If the workflow is struggling due to memory constraints, then this may be the place where the eventual out-of-memory error is raised. The need for extra memory management work can be reduced by reordering and grouping columns, optimizing and streamlining the type of data matrices, etc.

##### JPMML-Evaluator

The [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library is designed for making predictions on isolated data records. There is no functional need for vectorized math operations, C language libraries, or GPU/TPU acceleration.

The main selling point is quick and constant scoring times.
The cost of evaluating a batch of N data records equals N times the cost of evaluating the "average" data record.

All application scenarios are handled by the same data record-oriented API.
The elimination of batching, both conceptually and practically, guarantees strictly linear scaling properties, and gives JPMML-Evaluator the potential to outperform Scikit-Learn.

The PMML representation of data and pipelines uses high-level concepts.
It does not need to bother itself with data conversions from the rich real-life value space to a simplified numeric value space, nor data sparsity.

Last but not least, performing any kind of benchmarking on the Java/JVM platform assumes that the JVM has been properly "warmed up".
A JVM starts "cold". It gets warmer by monitoring the running application, and automatically identifying and optimizing performance-critical parts of the application code. A JVM is said to be "hot" when there is nothing left to optimize.

The warm-up procedure can be configured and guided by JVM command-line options. However, there are risks involved, because a misconfiguration may actually hamper the performance.

A JVM can be "warmed up" safely and easily by using the Java application in the intended way.
In the context of JPMML-Evaluator, this means evaluating data records until the average scoring time stabilizes.

The number of warm-up evaluations depends on the model type and complexity, but should be in the order of tens of thousands. Warm-up data records may be generated randomly based on the model schema, or be emebbed into the PMML document as the model verification dataset. 
The dataset does not need to consist of unique data records only. It is equally fine to iterate over a small but representative dataset multiple times.

### Materials and methods

##### Dataset

The "audit" dataset is loaded from a CSV document into a `pandas.DataFrame`.
Automatic data type detection and conversion results in two continuous integer features, one continuous float feature, and five categorical string features.
The cardinality of string features is low, ranging from 2 to 16 category levels.

The data is pre-processed minimally, just to make it comply with Scikit-Learn base expectations.
Continuous features are standardized in the linear model case, and left as-is in the decision tree ensemble case. Categorical features are one-hot encoded.

##### Modeling

This exercise uses two binary classification algorithms with vastly different properties.

First, the linear model case as implemented by [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model.

The `LogisticRegression.predict(X)` method can almost completely be expressed using vectorized math operations:

1. Compute the decision function (row vector).
2. Compute the probability distribution (row vector) by applying the sigmoid function to the decision function.
3. Select the class label that corresponds to the peak of the probability distribution.

As of Scikit-Learn version 0.24, the `predict(X)` method takes a shortcut and omits the second step (computation of the probability distribution) based on the premise that class label depends on the ranking, and not the magnitude, of decision function values. In other words, the application of sigmoid function is not necessary and should be avoided on a high cost-low merit basis (involves exponentiation).

Second, the decision tree ensemble case as implemented by [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) model.

The `RandomForestClassifier.predict(X)` method computes the probability distribution by averaging the probability distributions of its member decision trees, and then selects the class label that corresponds to its peak.

The evaluation of decision trees relies on conditional logic. For better performance, decision trees are molded as balanced binary trees, and implemented in Cython.

The only operation that could benefit from vectorization is the averaging of probability distributions.
However, when looking into the source code of `predict(X)` and `predict_proba(X)` methods, then it can be seen that computation is carried out differently. This suggests that the performance gain from vectorization would be insufficient to compensate for the performance loss from allocating a temporary `(n_trees, n_classes)` data matrix.

Fitted pipelines are first dumped in pickle data format using the `joblib` package, and then converted to the PMML representation using the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package.

##### Making comparisons

The objective is measuring the pure prediction time.

The model and the input data matrix are loaded outside of the main time measurement loop.

In Python, the data is loaded into a `DataFrame` object using the `pandas.read_csv(path)` utility function, which automatically takes care of column data type detection and conversion.
In Java, the data is loaded manually into a list of maps `java.util.List<java.util.Map<String, ?>>`; map keys are strings, map values are Java primitive values (pre-parsed from the raw string value).

For batch testing purposes it is necessary to draw new custom size datasets off the base dataset.
This is done using random sampling with replacement.
It is inevitable that Python and Java samples come to contain different data records in different order. However, from the benchmarking perspective, this difference should not be significant, because all data records are considered to be functionally and computationally equivalent.

### Results

##### Scikit-Learn

The Scikit-Learn experiment is about quantifying the effect of batch size.

Scoring times are reported for three setups to make the "contribution" of the transformers stage and the model stage explicit:

1. Full pipeline - `y = pipeline.predict(X)`
2. All the data transformations steps of the pipeline (without the final estimator step) - `Xt = Pipeline(pipeline.steps[0:-1]).transform(X)`
3. Only the final estimator step - `y = Pipeline([pipeline.steps[-1]]).predict(Xt)`.

By definition, the scoring time of setup #1 should equal the sum of scoring times of setups #2 and #3.

The setup can be controlled by supplying a third argument to the `benchmark.py` script (unset for setup #1; "transformers" and "model" for setups #2 and #3, respectively).

```
$ python benchmark.py <path to PKL file> <path to CSV file>
$ python benchmark.py <path to PKL file> <path to CSV file> "transformers"
$ python benchmark.py <path to PKL file> <path to CSV file> "model"
```

Logistic regression results:

| Configuration | Tranformers time (μs) | Model time (μs) | Pipeline time (μs) |
|---|---|---|---|
| 1000 * 1 | 6173.948 | 94.990 | 6268.938 |
| 1000 * 10 | 591.526 | 9.775 | 601.301 |
| 1000 * 100 | 60.858 | 1.001 | 61.859 |
| 10 * 1000 | 7.753 | 0.136 | 7.889 |
| 10 * 10000 | 2.216 | 0.028 | 2.244 |
| 1 * 100000 | 1.708 | 0.024 | 1.732 |

Random forest results:

| Configuration | Tranformers time (μs) | Model time (μs) | Pipeline time (μs) |
|---|---|---|---|
| 1000 * 1 | 5764.249 | 9583.312 | 15347.561 |
| 1000 * 10 | 585.291 | 1026.034 | 1611.325 |
| 1000 * 100 | 59.495 | 116.309 | 175.804 |
| 10 * 1000 | 7.392 | 18.779 | 26.171 |
| 10 * 10000 | 2.175 | 8.306 | 10.481 |
| 1 * 100000 | 1.656 | 7.387 | 9.043 |

The random error associated with time measurements is mitigated by running smaller-sized batches many times (and averaging their results).
Here, small batches are run 1000 times, medium batches 10 times, and large batches only once.
In the result tables (see above and below), run configurations are indicated using the formula `<number of runs> * <batch size>`. For example, the "1000 * 10" run configuration reads "create a batch of 10 data records, and invoke the `predict(X)` method 1000 times with it".

The scoring times between `benchmark.py` script re-runs do not fluctuate more than 5%, which is considered excellent for a simplictic Python command-line application.

Transformer times between LR and RF are roughly the same.
The cost of the extra `StandardScaler` step for LR is barely noticeable, because it relies on vectorized math operations.

Model times for LR are around 100X to 300X shorter than for RF.
On a member decision tree model basis, the evaluation of linear models (0.024 microsec) is up to 4X faster than the evaluation of decision trees (7.387 / 71 = 0.104 microsec).

Looking at transformer times vs. model times reveals that the LR pipeline is fully limited by the data pre-processing step, whereas the RF pipeline is more balanced.

Pipeline times indicate that Scikit-Learn is very sensitive to batch sizing.
With smaller batch sizes (1 to 1'000 data records) the average scoring time decreases 10X as the size of batch increases 10X.
For example, it takes roughly the same amount of time to evaluate a batch of 1 and a batch of 100 (eg. the same data record cloned the specified number of times).

##### JPMML-Evaluator in Java

The JPMML-Evaluator experiment is about quantifying the effect of JVM warm-up status.

The workflow involves evaluating a variable-size warm-up batch, followed by a fixed-size (100'000 data records) main batch.
The hypothesis is that warmer JVM should deliver shorter scoring times.
Different parts of JPMML-Evaluator bytecode get compiled to native code at different times. The warm-up function is therefore expected to exhibit many minor cliffs, not just one major cliff (ie. a global transition from the "cold" state to the "hot" state).

The `benchmark.Demo` command-line application takes four arguments.
The first two are the locations of PMML and CSV files in the local filesystem, respectively.
The third argument is the language environment emulation mode (one of "Java" or "Python"), and the fourth one is the size of the warm-up batch.

```
$ java -jar benchmark-executable-1.0-SNAPSHOT.jar <path to PMML file> <path to CSV file> "Java" <warm-up batch size>
```

Logistic regression results:

| Warm-up batch size | Time (μs) |
|---|---|
| 0 | 9.500 -- 12.720 |
| 1 | 9.250 -- 10.460 |
| 10 | 8.930 -- 10.730 |
| 100 | 9.690 -- 10.610 |
| 1000 | 8.990 -- 9.430 |
| 10000 | 5.820 -- 7.040 |
| 100000 | 4.190 -- 4.770 |

Random forest results:

| Warm-up batch size | Time (μs) |
|---|---|
| 0 | 98.230 -- 107.760 |
| 1 | 93.400 -- 101.260 |
| 10 | 97.410 -- 100.170 |
| 100 | 90.650 -- 95.110 |
| 1000 | 96.710 -- 97.450 |
| 10000 | 91.570 -- 95.760 |
| 100000 | 87.520 -- 90.230 |

The scoring times are tabulated in range notation, because they fluctuate considerably (up to 30%). The source of non-determinism is unclear.
The minimum and maximum values should adequately reflect the best and worst case performance, respectively. The expected value (ie. the mean of the distribution) remains unknown due to the small number of re-runs.

The JVM warm-up effect is well pronounced with LR, where scoring times differ ~2.5X between "cold" and "hot" states.
It can easily be overlooked with RF, where the this difference is roughly the same as the natural variance between scoring times (eg. the fastest "cold" scoring time is comparable to the slowest "hot" scoring time).

The warm-up functions appears to exhbit a similar shape in both cases.
There is a cliff between 0 and 1 batch sizes, then there is a slow and orderly descent between 1 and 10'000 batch sizes, followed by another cliff between 10'000 and 100'000 batch sizes.

The first cliff corresponds to JPMML-Evaluator internal lazy-loading/lazy-initialization work.
The descent corresponds to the JIT compilation of methods by the JVM. Methods are prioritized by their complexity and frequency of use. The JIT compilation starts with smaller and more popular methods, and proceeds until all JIT compilation-worthy methods have been processed.
The second cliff corresponds to reaching the "hot" state. The scoring time has reached a plateau, which will change only if the running JVM is perturbed with new information.

The JPMML-Evaluator library provides an `org.jpmml.evaluator.Evaluator#verify()` method, which evaluates the model with the embedded model verification dataset.
The above results show that model verification is good for crossing the first cliff, but is typically not adequate for reaching and crossing the second cliff.

Direct comparison of scoring times shows that JPMML-Evaluator outperforms Scikit-Learn with smaller batch sizes (below 1'000), but underperforms with larger batch sizes (over 10'000).
The best case is applying LR to a batch of 1, where JPMML-Evaluator outperforms by ~1000X (6268.938 / 4.770 = 1314).
The worst case is applying RF to a batch of 100'000, where JPMML-Evaluator underperforms by ~10X (9.043 / 90.230 = 0.1002).

It follows that Scikit-Learn and JPMML-Evaluator are complementary rather than competitive.

##### JPMML-Evaluator in Python: component analysis

The [`jpmml_evaluator`](https://github.com/jpmml/jpmml-evaluator-python) package provides a Python wrapper for the JPMML-Evaluator library.

The Java core is responsible for all heavy-lifting such as model loading and making predictions.
The Python driver to the Java core is responsible for the workflow coordination, data conversions and transfer.

Jumping back-and-forth between language environments is expensive.
In fact, the cost of calling Java from Python appears to be order(s) of magnitude higher than calling C from Python, which gives justification to devising rather complex schemes.

At the time of writing this (July 2021), the `jpmml_evaluator` package performs an evaluation round-trip as follows:

1. User/Python: unpack `DataFrame` to a list of dicts using the `DataFrame.to_dict(orient = "records")` method.
2. User/Python: dump arguments (list of dicts) in pickle data format.
3. System: (pass execution from Python to Java)
4. User/Java: load arguments (`List<Map<String, ?>>` from the pickle arguments dump.
5. User/Java: iterate over arguments, evaluate, collect results.
6. User/Java: dump results (`List<Map<String, ?>>`) in pickle data format.
7. System: (pass execution from Java back to Python)
8. User/Python: load results (list of dicts) from the pickle results dump.
9. User/Python: pack list of dicts to `DataFrame` using the `DataFrame.from_records(iterable of dicts)` function.

The JPMML-Evaluator-Python experiment is about quantifying the total cost of the Python data handler (steps #1, #2, #8 and #9), and its Java counterpart (steps #4 and #6).
The cost of running JPMML-Evaluator in Java (step #5) is already known separately.

All this data handler complexity exists for the sole purpose of mapping complex Python data structures to Java, and back.

The Python data science stack has the advantage that all Python tools and libraries can interface/communicate with each other directly using small number of standardized data structures.
If the data resides in shared memory, then Python application components can simply pass a memory reference to the data between each other.

Unfortunately, Java tools and libraries cannot interface with Pandas' data frames or Numpy arrays in a similar way.
The workaround is to deconstruct complex/language-specific data structures into simpler/language-agnostic data structures, which are adequately supported by both sides.

The `jpmml_evaluator` package communicates between Python and Java environments using the pickle protocol.
Since `DataFrame` is a complex data structure that does not have a Java equivalent, it is deconstructed into a list of dicts (data record-oriented API), which maps to a `List<Map<?, ?>>`.
Pickling uses native encoders on the Python side, and the [Pickle](https://github.com/irmen/pickle) library on the Java side.

###### Java data handler

The `benchmark.Main` application has "Python" language environment emulation mode, which adds arguments unpickling and results pickling work to the core "Java" mode.

```
$ java -jar benchmark-executable-1.0-SNAPSHOT.jar <path to PMML file> <path to CSV file> "Python" 100000
```

Logistic regression results:

| Configuration | Time (μs) |
|---|---|
| 1000 * 1 | 68.000 -- 104.000 |
| 1000 * 10 | 27.200 -- 43.800 |
| 1000 * 100 | 7.500 -- 7.900 |
| 10 * 1000 | 8.300 -- 11.000 |
| 10 * 10000 | 6.180 -- 6.450 |
| 1 * 100000 | 5.740 -- 5.980 |

Random forest results:

| Configuration | Time (μs) |
|---|---|
| 1000 * 1 | 115.000 -- 150.000 |
| 1000 * 10 | 90.900 -- 96.900 |
| 1000 * 100 | 92.240 -- 97.190 |
| 10 * 1000 | 90.300 -- 94.900 |
| 10 * 10000 | 88.910 -- 93.280 |
| 1 * 100000 | 88.900 -- 94.030 |

The pickling overhead for each batch size can be estimated by subtracting the JPMML-Evaluator scoring time from the tabulated scoring times.
The cost function appears to exhibit a 30 -- 60 microsecs fixed part, and a 1 -- 1.5 microsecs variable part.

###### Python data handler

The `benchmark.py` script has "Dummy" mode, where the predictions are made by a dummy (no-op) model.
To eliminate any systematic bias or error, this model returns a three-column data matrix (string class label, double probabilities of event and no-event) as is typical with binary classifiers.

Pickling is done using the pickle 2 protocol version.
Compression is turned off, because it would consume CPU cycles (cf. memory or IO bandwidth), which is the scarcest resource during benchmarking.

```
$ python benchmark.py <path to PKL file> <path to CSV file> "Dummy"
```

Dummy model results:

| Configuration | Time (μs) |
|---|---|
| 1000 * 1 | 217.917 |
| 1000 * 10 | 17.906 |
| 1000 * 100 | 2.360 |
| 10 * 1000 | 3.179 |
| 10 * 10000 | 0.684 |
| 1 * 100000 | 0.752 |

The cost function appears to exhibit a ~200 microsecs fixed part, and a 0.5 -- 1 microsecs variable part.

##### JPMML-Evaluator in Python: complete workflow analysis

The summation of component times gives the "user" workflow time, but there is an additional "system" workflow time, which corresponds to Python calling Java via some inter-process communication technology (steps #3 and #7).

The `benchmark.py` script has "JPMML/PyJNIus" and "JPMML/Py4J" modes for activating the [PyJNIus](https://github.com/kivy/pyjnius) and [Py4J](https://www.py4j.org/) backends, respectively.

```
$ python benchmark.py <path to PMML file> <path to CSV file> "JPMML/PyJNIus"
$ python benchmark.py <path to PMML file> <path to CSV file> "JPMML/Py4J"
```

Logistic regression results:

| Configuration | PyJNIus time (μs) | Py4J time (μs) |
|---|---|---|
| 1000 * 1 | 1453.429 -- 1602.972 | 1835.353 -- 2227.383 |
| 1000 * 10 | 167.379 -- 182.924 | 221.774 -- 249.934 |
| 1000 * 100 | 37.410 -- 39.164 | 46.621 -- 47.625 |
| 10 * 1000 | 25.152 -- 30.576 | 91.389 -- 104.549 |
| 10 * 10000 | 21.770 -- 23.646 | 38.932 -- 39.747 |
| 1 * 100000 | 22.058 -- 22.587 | 23.504 -- 23.762 |

Random forest results:

| Configuration | PyJNIus time (μs) | Py4J time (μs) |
|---|---|---|
| 1000 * 1 | 1683.092 -- 1866.688 | 2077.538 -- 2423.054 |
| 1000 * 10 | 283.360 -- 304.195 | 343.828 -- 357.657 |
| 1000 * 100 | 122.838 -- 130.278 | 139.218 -- 145.169 |
| 10 * 1000 | 104.193 -- 107.398 | 218.086 -- 231.611 |
| 10 * 10000 | 100.216 -- 101.682 | 121.297 -- 124.476 |
| 1 * 100000 | 100.881 -- 102.534 | 106.926 -- 110.737 |

The Python-to-Java backend overhead can be estimated for each batch size by subtracting the sum of data handler times from the tabulated scoring times.
It appears to be a fixed cost somewhere in the 1100 -- 1300 microsecs (PyJNIus) or 1500 -- 1700 microsecs (Py4J) range.

The PyJNIus backend has lower overhead than the Py4J backend.
However, the difference between the two is not that big in relative terms, which gives application developers freedom to work with either one.

### Main observations and conclusions

##### General

* Benchmarking should be carried out using a setup that mimics the intended production setup as closely as possible. Benchmarking unrelated transformer and model types with unrelated datasets is a fool's errand.

* Benchmarking is about determining the shape of all relevant cost functions. Once known, they can be analyzed individually (location and interpretation of cliffs), or overlayed with each other (location and interpretation of crossing points).

* Separating fixed costs (per batch) from variable costs (per each data record in the batch).
The ratio between fixed costs vs. variable costs dictates the minimum efficient batch size. Fixed costs typically dominate over variable costs, so the scoring time per data record decreases as the batch size increases. The optimal batch size is where the scoring times have reached a plateau. Pushing for even larger batch sizes may cause the scoring time per data record to begin increasing again due to new kinds of system-level fixed costs.

* Benchmarking using command-line applications assumes repeatability (ie. the consistency of results between identically configured runs). Python meets this requirement well (less than 5% variance), but Java not so much (10 -- 30% variance). The workaroud is to report results as ranges (observed minimum and maximum values over a number of runs).

* Benchmarking Java libraries and tools is sensitive towards the JVM warm-up status. The scoring times between "cold" and "hot" states may differ substantially. The warm-up needs to be complete relative to this part of the codebase that is being measured; unrelated parts may be left cold or lukewarm.

##### Scikit-Learn

* Scikit-Learn is designed around the batch processing idea, because it addresses both model training and model deployment concerns.

* Scikit-Learn is characterized by high fixed costs (5000 microsec per batch) and medium-to-low variable costs (2 to 10 microsec per data record).
Fixed costs are associated with managing Numpy arrays throughout the workflow. Variable costs are significant if the workflow deals with complex data type features (eg. strings, temporals, user-defined data types) that require transformation to numeric representation.
Variable costs are almost negligible when working with all-numeric data.

* Scikit-Learn benchmarks should measure transformers time and model time separately. Data pre-processing is often the rate-limiting step. The bottleneck can sometimes be relieved by simply reordering or restructuring transformer steps.

* Scikit-Learn can and will take advantage of vectorized math operations. However, the implementation resides at the C language layer, which limits its utility with smaller batch sizes.

##### JPMML-Evaluator in Java

* JPMML-Evaluator and its derivatives are designed exclusively around the model deployment concern.
In the PMML approach, the model training part is left to specialized ML frameworks such as Apache Spark, R or Scikit-Learn. The fitted pipelines are converted to the PMML representation using high-quality JPMML family conversion tools and libraries.
The foundation for success is already laid in the conversion stage, because JPMML converters perform sophisticated analyses and optimizations, which result in smallest, cleanest and most robust PMML documents.

* JPMML-Evaluator is characterized by zero fixed costs and medium variable costs (5 to 100 microsec per data record).
Variable costs depend more on the model type, and less on the model schema.

* JPMML-Evaluator computes the prediction by lazily evaluating one big computation graph.
The scoring path, and hence the scoring time, depends on the feature values of a data record. For some data records they are shorter, for others longer. The expected scoring time can be estimated by averaging the scoring times of all data records in a dataset.

* JPMML-Evaluator is oriented towards real-time and near real-time prediction (eg. stream scoring), where the data comes in sporadically, and has to be acted on immediately. However, it is also applicable to batch prediction, even though its peak throughput is limited due to the unavailability of vectorized math operations.

##### JPMML-Evaluator in Python

* JPMML-Evaluator-Python is characterized by high fixed costs (1000 to 1500 microsec per batch) and medium variable costs (20 to 100 microsec per data record).
The fixed costs stem from Python-to-Java inter-process communication, and are difficult to get around using the existing PyJNIus or Py4J backends.

* JPMML-Evaluator-Python is one or two orders of magnitude slower than JPMML-Evaluator, which limits its utility for real-time prediction. However, it is still competitive with Scikit-Learn when dealing with small batches.

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python scripts: [`train.py`]({{ "/resources/2021-08-04/train.py" | absolute_url }}) and [`benchmark.py`]({{ "/resources/2021-08-04/benchmark.py" | absolute_url }})
* Java application: [`benchmark.zip`]({{ "/resources/2021-08-04/benchmark.zip" | absolute_url }})