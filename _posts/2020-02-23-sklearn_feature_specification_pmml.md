---
layout: post
title: "Extending Scikit-Learn with feature specifications"
author: vruusmann
keywords: scikit-learn sklearn2pmml feature-domain data-categorical data-temporal data-missing
---

Predictive analytics applications must pay attention to "model-data fit", which means that a model can only be used if it is known to be relevant and applicable.

To illustrate, given a model object, one should be able to confidently answer questions like:

* Which features are needed?
* What is the domain of individual features? Are missing values supported?
* Is this combination of feature values typical or not?

Much of this (meta-)information is readily available during model training.
JPMML family conversion tools and libraries aim to capture, systematize and store it automatically, with minimal intrusion to existing workflows.

This blog post demonstrates how Scikit-Learn users should approach the "model-data fit" problematics.

## Overview ##

The Predictive Model Markup Language (PMML) defines data structures for representing most common model types.

Every model element holds the description of its data interface:

* Functional description of feature domains - the [`MiningSchema`](https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdElement_MiningSchema) element.
* Functional description of the prediction range - [`Targets`](https://dmg.org/pmml/v4-4-1/Targets.html#xsdElement_Targets) and [`Output`](https://dmg.org/pmml/v4-4-1/Output.html#xsdElement_Output) elements.
* Simple statistics about feature domains - the [`ModelStats`](https://dmg.org/pmml/v4-4-1/Statistics.html#xsdElement_ModelStats) element.
* Complex statistics about feature domains partitioned by the prediction range - the [`ModelExplanation`](https://dmg.org/pmml/v4-4-1/ModelExplanation.html#xsdElement_ModelExplanation) element.

Value preparation is a two-stage process.

In the first stage, the user value is converted to a PMML value according to the [`DataField`](https://dmg.org/pmml/v4-4-1/DataDictionary.html#xsdElement_DataField) element.

The user value is cast or parsed into the correct data type, restricted to the correct operational type (one of continuous, categorical or ordinal), and assigned to the value space (one of valid, invalid or missing).
The resulting PMML value can be regarded as a value in a three-dimensional space `<data type>-<operational type>-<value space type>`.

Consider the following `DataField` element:

``` xml
<DataField name="status" dataType="integer" optype="categorical">
  <Value value="1"/>
  <Value value="2"/>
  <Value value="3"/>
  <Value value="-999" property="missing"/>
</DataField>
```

Value conversions:

| Java value | PMML value | Explanation |
|------------|------------|-------------|
| `java.lang.String("1")` | `integer-categorical-valid` | Parseable, listed as valid |
| `java.lang.Integer(2)` | `integer-categorical-valid` | As-is, listed as valid |
| `java.lang.Double(3.0)` | `integer-categorical-valid` | Castable without loss of precision, listed as valid |
| `java.lang.String("one")` | `integer-categorical-invalid` | Not parseable |
| `java.lang.Integer(0)` | `integer-categorical-invalid` | As-is, not listed as valid |
| `java.lang.Double(3.14)` | `integer-categorical-invalid` | Not castable without loss of precision |
| `null` | `integer-categorical-missing` | Missing value |
| `java.math.BigDecimal("-999.000")` | `integer-categorical-missing` | Castable without loss of precision, listed as missing |

In the second stage, the PMML value undergoes one or more value space-dependent treatments according to the [`MiningField`](https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdElement_MiningField) element.

**Valid values pass by default**.
The domain of continuous values can be restricted by changing the value of the [`outliers`](https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdType_OUTLIER-TREATMENT-METHOD) attribute from `asIs` to `asMissingValues` or `asExtremeValues`, plus adding `lowValue` and `highValue` attributes.

**Invalid values _do not_ pass by default**, because the default value of the [`invalidValueTreatment`](https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdType_INVALID-VALUE-TREATMENT-METHOD) attribute is `returnInvalid`.

The behaviour where the model actively refuses to compute a prediction can be surprising to Scikit-Learn users.
However, this should be seen as a feature, not a bug, because the objetive is to inform upstream agents about data correctness and/or consistency issues (eg. feature drift) and prevent downstream agents from taking action on dubious results.

The model can be forced to accept invalid values by changing the value of the `invalidValueTreatment` attribute to `asIs`.
However, as every invalid value is "broken" in its own way, the computation may succeed or fail arbitrarily.

The recommended approach is to make the computation more controllable.

Invalid values may be replaced with a predefined valid value by changing the value of the `invalidValueTreatment` attribute to `asIs`, plus adding the `(x-)invalidValueReplacement` attribute (the `x-` prefix is required in PMML schema versions earlier than 4.4).
Alternatively, they may be replaced with a missing value by changing the value of the `invalidValueTreatment` attribute to `asMissing`.

**Missing values pass by default**.
By analogy with invalid value treatment, missing values can be rejected by changing the value of the [`missingValueTreatment`](https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdType_MISSING-VALUE-TREATMENT-METHOD) attribute to `(x-)returnInvalid`, or replaced with a predefined valid value by adding the `missingValueReplacement` attribute.

**Important**: The IEEE 754 constant NaN ("Not a Number") is assigned to invalid value space (not to missing value space).

## SkLearn2PMML domain decorators ##

The [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package provides several domain decorators for customizing the content of `DataField` and `MiningField` elements:

* `Domain(BaseEstimator, TransformerMixin)`
  * `ContinuousDomain`
  * `DicreteDomain`
    * `CategoricalDomain`
    * `OrdinalDomain`
  * `TemporalDomain`
    * `DateDomain`
    * `DateTimeDomain`
* `MultiDomain`

The PMML data type is derived from the Python data type, but it can be overriden using the `dtype` parameter.
The operational type is derived from the location of the subclass in class hierarchy.

If the training dataset contains masked missing values, then the value of the mask should be declared using the `missing_values` parameter.

For example, creating the stub of the above `DataField` element:

``` python
from sklearn2pmml.decoration import CategoricalDomain

domain = CategoricalDomain(dtype = int, missing_values = -999)
```

The valid value space cannot be set or overriden manually.
It is collected and stored automatically whenever the `Domain.fit(X, y)` method is called.

The outlier treatment, invalid value treatment and missing value treatment are PMML defaults, but they can be overriden using the corresponding parameters.
Parameter names and values are derived from PMML attribute names and values by changing the format from lower camelcase ("someValue") to lower underscore case ("some_value").

For example, making the default configuration explicit:

``` python
from sklearn2pmml.decoration import ContinuousDomain

domain = ContinuousDomain(outlier_treatment = "as_is", low_value = None, high_value = None, invalid_value_treatment = "return_invalid", invalid_value_replacement = None, missing_value_treatment = "as_is", missing_value_replacement = None)
```

The `Domain.transform(X)` method uses all this information to prepare the dataset exactly the same way as any standards-compliant PMML engine would do.

## Scikit-Learn examples ##

Domain decorators bring most value when working with heterogeneous datasets.

The simplest way to go about such workflows is to assemble a two-step pipeline, where the first step is either a `sklearn_pandas.DataFrameMapper` or `sklearn.compose.ColumnTransformer` meta-transformer for performing column-oriented data pre-processing work, and the second step is an estimator:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

import pandas

df = pandas.read_csv("audit.csv")

mapper = DataFrameMapper([
  (["Income"], None),
  (["Employment"], OneHotEncoder())
])

classifier = DecisionTreeClassifier()

pipeline = PMMLPipeline([
  ("mapper", mapper),
  ("classifier", classifier)
])
pipeline.fit(df, df["Adjusted"])
pipeline.verify(df.sample(n = 10))

sklearn2pmml(pipeline, "pipeline.pmml")
```

Some guiding principles to follow when introducing domain decorators:

* A domain decorator must be in the first position in the transformers list, because it can only be applied to input fields (`DataField` elements) and not to already encoded or transformed fields (`DerivedField` elements).
* A column should only be decorated once. If the same column is used multiple times, then the first occurrence should be decorated, and all the other occurrences should be left undecorated.
* It never hurts to be more specific and explicit. Default parameter values can be surprising at times.
* Domain decorators are supposed to help with assessing the "model-data fit" during model deployment. Are there any known differences (eg. encoding of missing values) between the training dataset and the deployment dataset(s)?

``` python
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain

mapper = DataFrameMapper([
  (["Income"], ContinuousDomain()),
  (["Employment"], [CategoricalDomain(), OneHotEncoder()])
])
```

Avoiding duplicate decorations:

``` python
from sklearn2pmml.decoration import Alias, MultiDomain
from sklearn2pmml.preprocessing import ExpressionTransformer, LookupTransformer

import numpy

employment_sector = {
  "Consultant" : "Private",
  "PSFederal" : "Public",
  "PSLocal" : "Public",
  "PSState" : "Public",
  "Private" : "Private",
  "SelfEmp" : "Private"
}

mapper = DataFrameMapper([
  (["Income"], ContinuousDomain()),
  (["Income", "Hours"], [MultiDomain([None, ContinuousDomain()]), Alias(ExpressionTransformer("X[0] / (X[1] * 52)", dtype = float), "Hourly_Income", prefit = True)]),
  (["Employment"], [CategoricalDomain(), OneHotEncoder()]),
  (["Employment"], [Alias(LookupTransformer(employment_sector, default_value = "Other"), "Employment_Sector", prefit = True), OneHotEncoder()])
])
```

In the above Python code, transformations have been grouped by input columns, whereas simple transformations ("Income", "Employment") have been moved in front of complex tranformations ("Hourly_Income", "Employment_Sector").
The "Hours" column does not make a standalone appearance.
It is decorated using the `MultiDomain` meta-decorator when the data enters the "Hourly_Income" transformers list.

Restricing the range of valid values:

``` python
mapper = DataFrameMapper([
  (["Income"], ContinuousDomain(outlier_treatment = "as_extreme_values", low_value = 2000, high_value = 400000)),
  (["Income", "Hours"], MultiDomain([None, ContinuousDomain(outlier_treatment = "as_missing_values", low_value = 0, high_value = 168, missing_value_treatment = "return_invalid", dtype = float)]), ...])
])
```

The "Income" column is restricted to `[2000, 400000]`.
The "Hours" column is restricted to `[0, 168]`, which represents the bounds of physical reality (number of hours in a week).
Any value outside that range is replaced with a missing value in order to trigger its rejection using the `returnInvalid` missing value treatment.

Customizing the treatment of invalid and missing values:

``` python
mapper = DataFrameMapper([
  (["Income"], ContinuousDomain(invalid_value_treatment = "as_is")),
  (["Employment"], [CategoricalDomain(invalid_value_treatment = "as_missing_value", missing_value_replacement = "Private"), OneHotEncoder()])
])
```

Decision trees are quite robust towards input values that were not present in the training dataset.
For example, continuous splits send the data record to the left or to the right by comparing the input value against the split threshold value.
These decisions do not carry any weight (eg. "weak left" vs. "strong right") that would depend on the distance between them.

Invalid and missing value spaces are often merged for convenience reasons.
No matter if the "Employment" column contains an invalid value or a missing value, it will be replaced with "Private" (the most frequent value in the training dataset).

## Resources ##

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
