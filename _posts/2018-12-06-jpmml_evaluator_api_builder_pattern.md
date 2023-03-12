---
layout: post
title: "JPMML-Evaluator: Upgrading from the Factory pattern to the Builder pattern"
author: vruusmann
keywords: jpmml-evaluator builder-pattern
---

Software grows and evolves over time. Even if the core concepts (types, relationships between them) stay the same, they acquire new properties and behaviour.

The [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library has relied on the [Factory (method-) pattern](https://en.wikipedia.org/wiki/Factory_method_pattern) to manage configuration and complexity.
However, this approach has reached its limits, as demonstrated by curiosities such as "factory of factories" and "factory method taking factories as arguments".

The biggest gripe with the Factory pattern is its statelessness and poor extensibility. All the object creation and configuration work is captured inside a single factory method, which can be interacted with via a long list of formal parameters.
The Factory pattern might be a great fit for dynamic programming languages, but not for the Java programming language. For example, in Java, every added/updated/removed formal parameter results in a breaking API change, and it is not possible to assign default values to them. Moreover, base factory methods are rather difficult to override with more specialized ones.

The way forward is the [Builder pattern](https://en.wikipedia.org/wiki/Builder_pattern).
A builder object proposes a sensible default configuration, which can be queried and incrementally updated using accessor methods. The object creation work happens inside the no-arguments `#build()` method. The builder object as a whole is typically reusable and serializable.

## Moving from ModelEvaluatorFactory to ModelEvaluatorBuilder ##

JPMML-Evaluator versions 1.2.0 through 1.4.3 have been recommending the following boilerplate code for creating a `ModelEvaluator` object:

``` java
PMML pmml = ...;

ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();

ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
```

JPMML-Evaluator version 1.4.4 proposes the following boilerplate code for the same:

``` java
PMML pmml = ...;

ModelEvaluatorBuilder modelEvaluatorBuilder = new ModelEvaluatorBuilder(pmml);

ModelEvaluator<?> modelEvaluator = modelEvaluatorBuilder.build();
```

The `org.jpmml.evaluator.ModelEvaluatorBuilder` class holds two kinds of state.
First, the "primary state" is concerned with the PMML class model. Its `#pmml` and `#model` fields are initialized during object construction, and cannot be modified (at least via the public API) later on.
Second, the "secondary state" is concerned with control and configuration. There are a number of factory-related fields that can be set and reset using mutator methods.

The `ModelEvaluatorBuilder` class is not thread-safe.
However, it is possible to effectively "freeze" its instances by casting them to instances of the `EvaluatorBuilder` super interface, which does not expose any mutator methods:

``` java
EvaluatorBuilder evaluatorBuilder = new ModelEvaluatorBuilder(pmml)
  .setValueFactoryFactory(ReportingValueFactoryFactory.newInstance());

Evaluator evaluator = evaluatorBuilder.build();
```

The `EvaluatorBuilder#build()` method can be invoked any number of times. It is up to the implementation class to decide if it creates a new `Evaluator` object for each invocation, or caches and keeps returning the same `Evaluator` object.

## Moving from ModelEvaluatorBuilder onward ##

In most common application scenarios, there is more application code involved in unmarshalling a PMML byte stream or a PMML file into into a live `org.dmg.pmml.PMML` object, than turning the latter into an `Evaluator` object.

The `LoadingModelEvaluatorBuilder` class aims to reduce complexity in this area. It is designed as a `ModelEvaluatorBuilder` subclass. The main difference is that the "primary state" is not set via constructor(s), but in a delayed manner via dedicated `#load(java.io.InputStream)` and `#load(java.io.File)` loader methods:

``` java
InputStream pmmlIs = ...;

EvaluatorBuilder evaluatorBuilder = new LoadingModelEvaluatorBuilder()
  .load(pmmlIs);

Evaluator evaluator = evaluatorBuilder.build();
```

The "secondary state" is extended by control and configuration over SAX parsing, XML schema validation, PMML class model object modification and optimization, etc.

For example, creating a `LoadingModelEvaluatorBuilder` object for the production environment:

``` java
// Load the XML schema definition (XSD) from the JPMML-Model library (jar:///pmml.xsd)
javax.xml.validation.Schema jpmmlSchema = org.jpmml.model.JAXBUtil.getSchema();

// Ordering of visitors by family - optimize, intern, finalize
VisitorBattery visitorBattery = new VisitorBattery();
visitorBattery.addAll(new ElementOptimizerBattery());
visitorBattery.addAll(new AttributeInternerBattery());
visitorBattery.addAll(new ElementInternerBattery());
visitorBattery.addAll(new ListFinalizerBattery());

LoadingModelEvaluatorBuilder loadingModelEvaluatorBuilder = new LoadingModelEvaluatorBuilder()
  // Activate XML schema validation
  .setSchema(jpmmlSchema)
  // Discard SAX Locator information (line and column numbers for PMML elements)
  // This can reduce memory consumption up to 25%
  .setLocatable(false)
  // Intern and optimize PMML elements and attributes
  // This can reduce memory consumption up to 50%, and increase evaluation speeds up to several hundred percent
  .setVisitors(visitorBattery);
```

Just like its super class, the `LoadingModelEvaluatorBuilder` class is not thread-safe either. A good workaround for parallel-processing applications is to maintain a template object, and make copies of it (one per worker thread) using the `Cloneable#clone()` method:

```
File pmmlFile = ...;

EvaluatorBuilder evaluatorBuilder = loadingModelEvaluatorBuilder.clone()
  .load(pmmlFile);

Evaluator evaluator = evaluatorBuilder.build();
```
