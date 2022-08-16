---
layout: post
title: "JPMML-Evaluator: Tracing and reporting machine learning model predictions"
author: vruusmann
keywords: jpmml-model jpmml-evaluator jpmml-evaluator-python mathml builder-pattern testing reporting
---

There are numerous application scenarios which require an ability to "look into" a model to understand how a particular prediction was computed. They range from low-stakes applications scenarios such as tracing and debugging misbehaving models, to high-stakes ones such as generating reports for models that are making life-changing decisions.

Most ML frameworks completely overlook this need. For example, Scikit-Learn logistic regression models expose `predict(X)` and `predict_proba(X)` methods, which return plain numeric predictions. The only way to understand how a particular number was computed (eg. active terms and their coefficients, the family and parameterization of the link function) is to open the source code of the logistic regression model class in a text editor, and parse/interpret the body of the predict method line-by-line. However, if the model operates on a transformed feature space, and the ML framework itself uses low-level abstractions for feature representation (eg. string features are transformed to binary vectors), then it is virtually impossible for a casual observer to make any sense of it all.

This problem has an easy two-step solution. First, the model or pipeline should be converted from the low-level ML framework representation to the high-level Predictive Model Markup Language (PMML) representation, which makes it human-readable and -interpretable in the original feature space. Second, all the tracing and reporting work should be automated using a PMML engine.

### Reporting Java API

The [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library is probably the most capable and versatile PMML engine for the Java/JVM platform. It provides different API levels and hooks for interacting with deployed models, including a special-purpose Value API for capturing all operations that are made when computing a prediction.

The appropriate Value API can be activated using the `org.jpmml.evaluator.ModelEvaluatorBuilder#setValueFactoryFactory(org.jpmml.evaluator.ValueFactoryFactory)` method. For example, creating two `Evaluator` objects based on the same in-memory `org.dmg.pmml.PMML` object:

``` java
PMML pmml = ...;

ModelEvaluatorBuilder evaluatorBuilder = new ModelEvaluatorBuilder(pmml);

// Uses the default Value API
Evaluator defaultEvaluator = evaluatorBuilder.build();

// Activate the reporting Value API
evaluatorBuilder = evaluatorBuilder
  .setValueFactoryFactory(ReportingValueFactoryFactory.newInstance());

// Uses the reporting Value API
Evaluator reportingEvaluator = evaluatorBuilder.build();
```

The reporting Value API captures the computation in the Mathematical Markup Language (MathML) representation. MathML is an XML dialect, which can be rendered as image, or translated to other data formats and representations such as LaTeX, or R and Python language expressions.

When the reporting Value API is activated, then target field value(s) shall be complex objects that implement the `org.jpmml.evaluator.HasReport` marker interface. This interface declares a sole `HasReport#getReport()` method, which gives access to the live `org.jpmml.evaluator.Report` object. The `Report` class is polymorphic, and has several specialized implementation classes available. The simplest way to obtain the final MathML string is to invoke the `org.jpmml.evaluator.ReportUtil#format(Report)` utility method:

``` java
Map<FieldName, ?> arguments = ...;
Map<FieldName, ?> results = reportingEvaluator.evaluate(arguments);

List<TargetField> targetFields = reportingEvaluator.getTargetFields();
for(TargetField targetField : targetFields){
  Object targetValue = results.get(targetField.getName());
  System.out.println("target=" + EvaluatorUtil.decode(targetValue));

  // The target field (aka label) of regression and classification models
  if(targetValue instanceof HasReport){
    HasReport hasReport = (HasReport)targetValue;

    Report report = hasReport.getReport();
    if(report != null){
      System.out.println("target=" + ReportUtil.format(report));
    }
  } // End if

  // The target field of probabilistic classification models
  if(targetValue instanceof HasProbability){
    HasProbability hasProbability = (HasProbability)targetValue;

    Set<String> targetCategories = hasProbability.getTargetCategories();
    for(String targetCategory : targetCategories){
      Double probability = hasProbability.getProbability(targetCategory);
      System.out.println("probability(" + targetCategory + ")=" + probability);

      Report probabilityReport = hasProbability.getProbabilityReport(targetCategory);
      if(probabilityReport != null){
        System.out.println("probability(" + targetCategory + ")=" + ReportUtil.format(probabilityReport));
      }
    }
  }
}
```

### Reporting PMML vendor extension

After successfully designing and implementing the reporting Value API, the authors made a suggestion to Data Mining Group (DMG.org) that the PMML standard should incorporate similar functionality in the form of a [`report` result feature](http://mantis.dmg.org/view.php?id=184). Unfortunately, DMG.org decided against doing so, which leaves everything into the status of a vendor extension.

A reporting `OutputField` element has the following attributes:

* `name` - The name of the output field. A good convention is to wrap the name of the base output field as `reporting(<output field name>)`.
* `dataType` and `optype` - Fixed as `string` and `categorical`, respectively.
* [`feature`](http://dmg.org/pmml/v4-4-1/Output.html#xsdType_RESULT-FEATURE) - Fixed as `x-report`. The "x-" prefix to the attribute value indicates that this is a vendor extension.
* `x-reportField` - The name of the base output field. Again, the "x-" prefix to the attribute name indicates that this is a vendor extension.

For example, enhancing a binary classification model to extract probability calculation reports for the "event" and "no-event" target categories:

``` xml
<Output>
  <!-- Base output fields -->
  <OutputField name="probability(event)" dataType="double" optype="continuous" feature="probability"/>
  <OutputField name="probability(no event)" dataType="double" optype="continuous" feature="probability"/>
  <!-- Reporting output fields -->
  <OutputField name="report(probability(event))" dataType="string" optype="categorical" feature="x-report" x-reportField="probability(event)"/>
  <OutputField name="report(probability(no event))" dataType="string" optype="categorical" feature="x-report" x-reportField="probability(no event)"/>
</Output>
```

The ordering of `OutputField` elements is not significant, except for the common sense requirement that the declaration of the base output field must precede the declaration of the reporting output field that references it.

### Python workflow

Training a minimalistic XGBoost model for the "audit" dataset:

``` python
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

import pandas

df = pandas.read_csv("audit.csv")

pipeline = PMMLPipeline([
  ("mapper", DataFrameMapper(
    [(cat_column, [CategoricalDomain(with_statistics = False), LabelBinarizer()]) for cat_column in ["Education", "Employment", "Marital", "Occupation", "Gender"]] +
    [(cont_column, [ContinuousDomain(with_statistics = False)]) for cont_column in ["Age", "Income"]]
  )),
  ("classifier", XGBClassifier(objective = "binary:logistic", n_estimators = 17, seed = 13))
])
pipeline.fit(df, df["Adjusted"])

sklearn2pmml(pipeline, "XGBoostAudit.pmml")
```

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package encodes this XGBoost model in the form of a two-segment model chain. The first segment is the "booster", which sums the predictions of 17 member decision tree models. The second segment is the "sigmoid function", which transforms the boosted value to a pair of probability values.

The generation of reporting `OutputField` elements could be controlled using a special-purpose conversion option. However, for as long as it is not available, or when working with existing and/or third-party PMML documents, then they need to be generated manually.

The newly generated PMML document `XGBoostAudit.pmml` is copied into `XGBoostAudit-reporting.pmml`, and modified in a text editor in the following way:

``` xml
<MiningModel>
  <!-- Omitted MiningSchema element -->
  <Segmentation multipleModelMethod="modelChain">
    <!-- "booster" -->
    <Segment id="1">
      <True/>
      <MiningModel functionName="regression" x-mathContext="float">
        <!-- Omitted MiningSchema element -->
        <Output>
          <OutputField name="xgbValue" optype="continuous" dataType="float" feature="predictedValue" isFinalResult="false"/>
          <OutputField name="report(xgbValue)" optype="categorical" dataType="string" feature="x-report" x-reportField="xgbValue"/>
        </Output>
        <!-- Omitted Segmentation element -->
      </MiningModel>
    </Segment>
    <!-- "sigmoid function" -->
    <Segment id="2">
      <True/>
      <RegressionModel functionName="classification" normalizationMethod="logit" x-mathContext="float">
        <MiningSchema>
          <MiningField name="Adjusted" usageType="target"/>
          <MiningField name="xgbValue"/>
          <MiningField name="report(xgbValue)"/>
        </MiningSchema>
        <Output>
          <OutputField name="ref(report(xgbValue))" optype="continuous" dataType="string" feature="transformedValue">
            <FieldRef field="report(xgbValue)"/>
          </OutputField>
          <OutputField name="probability(0)" optype="continuous" dataType="float" feature="probability" value="0"/>
          <OutputField name="probability(1)" optype="continuous" dataType="float" feature="probability" value="1"/>
          <OutputField name="report(probability(0))" optype="categorical" dataType="string" feature="x-report" x-reportField="probability(0)"/>
          <OutputField name="report(probability(1))" optype="categorical" dataType="string" feature="x-report" x-reportField="probability(1)"/>
        </Output>
        <RegressionTable intercept="0.0" targetCategory="1">
          <NumericPredictor name="xgbValue" coefficient="1.0"/>
        </RegressionTable>
        <RegressionTable intercept="0.0" targetCategory="0"/>
      </RegressionModel>
    </Segment>
  </Segmentation>
</MiningModel>
```

According to the PMML specification, the results provided from the model chain are the results of the last active segment. The results from earlier active segments must be explicitly propagated.
For example, the value of the "report(xgbValue)" output field stays in "booster" scope by default. It needs to be imported from "booster" scope into "sigmoid function" scope using a `MiningField` element, and then re-exported as "ref(report(xgbValue))" using an `OutputField` element.

The [`jpmml_evaluator`](https://github.com/jpmml/jpmml-evaluator-python) package provides a Python wrapper for the JPMML-Evaluator library. It enables quick PMML validation and evaluation work, without writing a single line of Java application code.

Creating a verified PMML engine, and evaluating the first row of the "audit" dataset:

``` python
from jpmml_evaluator import make_evaluator
from jpmml_evaluator.py4j import launch_gateway, Py4JBackend

# Launch Py4J server
gateway = launch_gateway()

backend = Py4JBackend(gateway)

evaluator = make_evaluator(backend, "XGBoostAudit-reporting.pmml", reporting = True) \
  .verify()

arguments = {
  "Age" : 38,
  "Employment" : "Private",
  "Education" : "College",
  "Marital" : "Unmarried",
  "Occupation" : "Service",
  "Income" : 81838,
  "Gender" : "Female",
  "Deductions" : False,
  "Hours" : 72
}
print(arguments)

results = evaluator.evaluate(arguments)
print(results)

# Shut down Py4J server
gateway.shutdown()
```

The result is a `dict` object with six items:

``` python
{
  "Adjusted" : 0,
  "ref(report(xgbValue))" : "<math><apply><plus/><cn>-0.17968129</cn><cn>-0.16313718</cn><cn>-0.15570186</cn><cn>-0.14460582</cn><cn>-0.13957009</cn><cn>-0.12399502</cn><cn>-0.12931323</cn><cn>-0.12063908</cn><cn>-0.11913132</cn><cn>-0.11191886</cn><cn>-0.11103699</cn><cn>-0.110534586</cn><cn>-0.101101674</cn><cn>-0.104573585</cn><cn>-0.09484299</cn><cn>-0.095708475</cn><cn>-0.09683787</cn></apply></math>",
  "probability(0)": 0.8911294,
  "probability(1)": 0.1088706,
  "report(probability(0))": "<math><apply><minus/><cn>1</cn><cn>0.1088706</cn></apply></math>",
  "report(probability(1))": "<math><apply><apply><inverse/><ci>logit</ci></apply><apply><plus/><apply><times/><cn>1.0</cn><cn>-2.1023297</cn></apply><cn>0.0</cn></apply></apply></math>"
}
```

The above report shows that the boosted value -2.1023297 is obtained by summing 17 member values that range from -0.17968129 to -0.09484299. As is typically the case with gradient boosting methods, the magnitude of member values decreases with each iteration.
The probability of the positive scenario 0.1088706 is obtained by applying the inverse of the logit function to -2.1023297. The probability of the negative scenario 0.8911294 is obtained by subtracting the probability of the positive scenario from 1.

### Resources

* "Audit" dataset: [`audit.csv`]({{ site.baseurl }}/assets/data/audit.csv)
* Python scripts: [`train.py`]({{ site.baseurl }}/assets/2019-02-26/train.py) and [`report.py`]({{ site.baseurl }}/assets/2019-02-26/report.py)
* Reporting PMML document: [`XGBoostAudit-reporting.pmml`]({{ site.baseurl }}/assets/2019-02-26/XGBoostAudit-reporting.pmml)