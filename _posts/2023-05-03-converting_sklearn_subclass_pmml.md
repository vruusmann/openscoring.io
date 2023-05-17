---
layout: post
title: "Converting customized Scikit-Learn estimators to PMML"
author: vruusmann
keywords: scikit-learn sklearn2pmml jpmml-sklearn
---

Scikit-Learn transformers and models are easily customizable via subclassing.

The simplest use case is "freezing" parameterizations.
For example, defining a `MultinomialClassifier` class based on the `sklearn.linear_model.LogisticRegression` class:

``` python
from sklearn.linear_model import LogisticRegression

class MultinomialClassifier(LogisticRegression):

  def __init__(self, **params):
    super(MultinomialClassifier, self).__init__(penalty = None, multi_class = "multinomial", **params)
```

Scikit-Learn is not concerned with class identities.
A newly defined class can be plugged into existing workflows without changing or configuring anything extra:

``` python
from sklearn.datasets import load_iris
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)

classifier = MultinomialClassifier(solver = "lbfgs")

pipeline = PMMLPipeline([
  ("classifier", classifier)
])
pipeline.fit(iris_X, iris_y)

# Raises RuntimeError
sklearn2pmml(pipeline, "MultinomialIris.pmml")
```

However, any attempt to convert the fitted pipeline object to the PMML representation using the `sklearn2pmml` package shall fail with a runtime error stating that the `__main__.MultinomialClassifier` class is not a supported. 

## SkLearn2PMML ##

The `sklearn2pmml.sklearn2pmml` utility function dumps the pipeline object in Pickle data format into a temporary file in the local filesystem, and then executes the JPMML-SkLearn command-line application to perform the actual conversion work.

### The registry of estimator-to-converter class mappings

The JPMML-SkLearn library operates in the Java environment.
When it loads a Pickle file, then it gets presented with Python class instances in their low-level serialized form that consists of a fully-qualified name (FQN) and a map of attributes.

The JPMML-SkLearn library does not have a mechanism for resolving Python class names to full-blown class definitions, which would allow (re)constructing Python class hierarchies, and checking if and how different Python classes relate to one another.
As a workaround, it maintains a registry of supported Python classes.

This registry can be loaded using the `sklearn2pmml.load_class_mapping` utility function.
It is returned in `dict` representation, where the keys are Scikit-Learn transformer or model class names, and the values are the corresponding JPMMl-SkLearn converter class names:

``` python
from sklearn2pmml import load_class_mapping

mapping = load_class_mapping()
print(len(mapping))
```

#### Entry keys

At the time of writing this (May 2023), the registry contains approximately 400 entries, which can be categorized into 250 "main" entries (top-level estimator classes) and 150 "helper" entries (aliases of estimator classes, nested and helper classes).

For example, the `sklearn.linear_model.LogisticRegression` model has two registrations, because its Python class has been moved from one module to another over the years.
The `sklearn.linear_model.logistic.LogisticRegression` key targets Scikit-Learn versions [0.16](https://github.com/scikit-learn/scikit-learn/blob/0.16.0/sklearn/linear_model/logistic.py) through [0.21](https://github.com/scikit-learn/scikit-learn/blob/0.21.0/sklearn/linear_model/logistic.py).
The `sklearn.linear_model._logistic.LogisticRegression` key (note the added underscore prefix!) targets Scikit-Learn versions [0.22](https://github.com/scikit-learn/scikit-learn/blob/0.22/sklearn/linear_model/_logistic.py) and newer.

**Important**: All Python classes must be referenced by their fully-qualified names.
When unsure, consider formatting the Python class using the `sklearn2pmml.util.fqn` utility function.

Querying the key set of the registry to see if some Scikit-Learn estimator class is currently supported or not:

``` python
from sklearn2pmml.util import fqn

print(fqn(LogisticRegression))
print(fqn(MultinomialClassifier))

assert "sklearn.linear_model.LogisticRegression" not in mapping, "Has no mapping for LR public name"
assert fqn(LogisticRegression) in mapping, "Has mapping for LR fully-qualified name"
assert fqn(MultinomialClassifier) not in mapping, "Has no mapping for MC fully-qualified name"
```

#### Entry values

An estimator class is supported if there is a JPMML-SkLearn converter class mapped to it.

The package structure of JPMML-SkLearn draws inspiration from Scikit-Learn, but there are no clear rules for translating class names between the two libraries.

It can be found that multiple Scikit-Learn models are often mapped to the same JPMML-SkLearn converter.
For example, over 20 linear regressor classes are mapped to the `sklearn.linear_model.LinearRegressor` class.
This is possible, because these classes have differing `fit(X, y)` methods, but identical fitted state as captured by `coef_` and `intercept_` attributes.

Looking up and mapping JPMML-SkLearn converter classes:

``` python
import inspect

def make_class_mapping(cls, templateCls = None):

  if not templateCls:
    templateCls = inspect.getmro(cls)[1]

  pyClazzName = fqn(cls)
  javaConverterClazzName = mapping[fqn(templateCls)]

  return {
    pyClazzName : javaConverterClazzName
  }

mapping_cust = make_class_mapping(MultinomialClassifier)
print(mapping_cust)
```

For newly defined classes, the lookup should iterate over parent classes in the Python method resolution order (MRO) in order to find the most specific "template" possible.
However, nothing stops the data scientist from experimenting with alternative JPMML-SkLearn converter classes.

There are a number of classes that are useful way beyond their original scope.
For example, if the pipeline contains some (pseudo-)transformer for which there is no PMML representation, then it is possible to make the JPMML-SkLearn library skip over it by mapping it to the identity transformer:

``` python
from sklego.preprocessing import IdentityTransformer

mapping_skipover = make_class_mapping(SomeTroublemakerTransformer, IdentityTransformer)
```

### Registry expansion

The `load_class_mapping` utility function is provided only for information purposes.
Any changes to the returned `dict` object will be ignored.

#### Classpath

The JPMML-SkLearn library loads this registry independently from the Java Archive (JAR) files on its classpath.

The classpath has two parts.
First, the "package classpath" part is a list of JPMML-SkLearn library JAR files that are included in the `sklearn2pmml` package as Python package data.
Second, the "user classpath" part is a list of JAR files in the local filesystem that the end user wants to append to it.

Inspecting the classpath:

``` python
from sklearn2pmml import _classpath

classpath = _classpath(user_classpath = [])
print(classpath)
```

The order of classpath elements is not significant.

#### Class mapping JAR

The registry can be expanded with new entries by adding new class mapping JAR files to the user classpath.

Technically, a so-called class mapping JAR file is a JAR file that contains a [Java properties](https://docs.oracle.com/javase/8/docs/api/java/util/Properties.html)-style `META-INF/sklearn2pmml.properties` class mapping file.

The first version of a class mapping JAR file can be created using the `sklearn2pmml.make_class_mapping_jar` utility function.
If need be, then it can then be manipulated further using Python's built-in `zipfile` module, because JAR files are nothing but decorated ZIP files.

``` python
from sklearn2pmml import make_class_mapping_jar

make_class_mapping_jar(mapping_cust, "multinomial.jar")

mapping = load_class_mapping(user_classpath = ["multinomial.jar"])

assert fqn(MultinomialClassifier) in mapping, "Has mapping for MC fully-qualified name"
```

Re-attempting the PMML conversion with the user classpath:

``` python
sklearn2pmml(pipeline, "MultinomialIris.pmml", user_classpath = ["multinomial.jar"])
```

The operation completes successfully this time.

#### Classpath maintenance

The life-cycle of a class mapping JAR file depends on the application scenario.

For as long as the `MultinomialClassifier` class remains nothing but a local experiment, it is perfectly fine to create and destroy the supporting class mapping JAR file on demand.
However, when the `MultinomialClassifier` class graduates from the incubator, then the same must happen to its supporting resources.

Third-party libraries are recommended to adopt the `sklearn2pmml` package resource loading conventions.
In brief, all user JAR files should be included in the package as Python package data, and they should be listable using an utility function such as `<mypackage>.resources._package_classpath`:

``` python
from mypackage.resources import _package_classpath

sklearn2pmml(..., user_classpath = _package_classpath())
```

## Resources ##

* Python script: [`train.py`]({{ "/resources/2023-05-03/train.py" | absolute_url }})