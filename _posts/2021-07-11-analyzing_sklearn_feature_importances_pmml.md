---
layout: post
title: "Analyzing Scikit-Learn feature importances via PMML"
author: vruusmann
keywords: scikit-learn sklearn2pmml feature-importance
---

## Terminology

**Feature importance** reflects which features are considered to be significant by the ML algorithm during model training.

Feature importance is a relative metric. It is often expressed on the percentage scale.
The main application area is ranking features, and providing guidance for further feature engineering and selection work.
For example, the cost and complexity of models can be reduced by gradually eliminating low(est)-importance features from the training dataset.

Feature importance is sometimes confused with feature impact.

**Feature impact** reflects which features and to which extent contribute towards the prediction when the fitted model is executed.

Feature impact is calculated by substituting feature values into the model equation, and aggregating the partial scores of model terms feature-wise.
This calculation is applicable to all data records, irrespective of their origin (ie. training, validation and testing datasets).

## Scikit-Learn

Some model types have built-in feature importance estimation capabilities.
For example, decision tree and decision tree ensemble models declare a `feature_importances_` property that yields Gini Impurities.
Similarly, it is not formalized as a linear model property, but all seasoned data scientists know that the beta coefficients of a linear model act as surrogate feature importances (assuming standardized data).

Scikit-Learn version 0.24 and newer provide the [`sklearn.inspection.permutation_importance`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) utility function for calculating permutation-based importances for all model types.

The estimation is feasible in two locations.

First, estimating the **importance of raw features** (data before the first data pre-processing step).
Indicates which columns of a structured data source such as a CSV document or a relational database are critical for success.

``` python
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipelin

pipeline = make_pipeline(transformer, classifier)
pipeline.fit(X, y)

# Perform PI calculation using the data as it entered the pipeline
imp_pipeline = permutation_importance(pipeline, X, y, random_state = 13)
print(imp_pipeline.importances_mean)
```

Second, estimating the **importance of fully-developed features** (data after the last data pre-processing step).
Indicates how to improve feature engineering and selection work.
For example, optimizing feature encodings, exploring and generating feature interactions, deriving custom features.

``` python
transformer.fit(X, y)

# Transform raw features to fully-developed features
Xt = transformer.transform(X)

classifier.fit(Xt, y)

# TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
Xt = Xt.todense()

# Perform PI calculation using the data as it entered the classifier
imp_classifier = permutation_importance(classifier, Xt, y, random_state = 13)
print(imp_classifier.importances_mean)
```

## Pickle

The [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) library can only work with this state of a Python object that is serializable in pickle data format.
A Python property does not have a persistent state. The workaround is to transfer its value into a new regular Python attribute.

By convention, the JPMML-SkLearn library checks if the Python pipeline or model object has a `pmml_feature_importances_` attribute (the `pmml_` prefix prepended to the standard `feature_importances_` attribute name).
If it does, then it is expected to hold a Numpy array of shape `(n_features, )`.

Exposing decision tree feature importances:

``` python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state = 13)
pipeline = make_fit_pmml_pipeline(dtc)

dtc.pmml_feature_importances_ = dtc.feature_importances_

sklearn2pmml(pipeline, "DecisionTreeAudit.pmml")
```

In case of ensemble models there could be feature importances available at different aggregation levels.

Exposing decision tree ensemble feature importances, first at the root model level, and then at the member model level:

``` python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 31, random_state = 13)
pipeline = make_fit_pmml_pipeline(rfc)

rfc.pmml_feature_importances_ = rfc.feature_importances_
for rfc_dtc in rfc.estimators_:
  rfc_dtc.pmml_feature_importances_ = rfc_dtc.feature_importances_

sklearn2pmml(pipeline, "RandomForestAudit.pmml")
```

Attaching custom feature importances to a PMML pipeline:

``` python
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
pipeline = make_fit_pmml_pipeline(lr, standardize = True)

result = permutation_importance(pipeline, audit_X, audit_y, random_state = 13)
pipeline.pmml_feature_importances_ = result.importances_mean

sklearn2pmml(pipeline, "LogisticRegressionAudit.pmml")
```

## PMML

The [`MiningField`](https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdElement_MiningField) element specifies an `importance` attribute for recording field importance values.

The PMML term "field" is incompatible with the Scikit-Learn term "feature".
The former corresponds to **raw feature** (data before the first pre-processing step), whereas the latter corresponds to **fully-developed feature** (data after the last pre-processing step).
They are functionally equivalent only when the pipeline does not contain any data pre-processing steps.

Thanks to the relative nature of Scikit-Learn feature importances it is possible to reduce them to PMML field importances via simple summation.
If a feature is derived from two or more columns, then its importance value is split between them in equal proportions.

The JPMML family of conversion tools and libraries preserves native feature importance information by attaching an `X-FeatureImportances` extension element to the [`MiningSchema`](https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdElement_MiningSchema) element. 
This extension element contains a table of feature names mapped to their importance values.
In the table header, there is a quick summary (the number and the sum of non-zero importance values, extreme non-zero importance values, etc.) to facilitate data parsing and interpreration.

For example, the PMML representation of feature importances for the "DecisionTreeAudit" case is the following:

``` xml
<MiningSchema>
  <Extension name="X-FeatureImportances">
    <InlineTable>
      <Extension name="numberOfImportances" value="49"/>
      <Extension name="numberOfNonZeroImportances" value="36"/>
      <Extension name="sumOfImportances" value="1.0"/>
      <Extension name="minImportance" value="2.0667387553303666E-4"/>
      <Extension name="maxImportance" value="0.20155442366598394"/>
      <!-- Omitted PMML content -->
      <row>
        <data:name>Hours</data:name>
        <data:importance>0.07440333502077028</data:importance>
      </row>
      <row>
        <data:name>Income</data:name>
        <data:importance>0.14371484745257854</data:importance>
      </row>
      <row>
        <data:name>Hourly_Income</data:name>
        <data:importance>0.15704030730211904</data:importance>
      </row>
      <!-- Omitted PMML content -->
      <row>
        <data:name>Marital=Absent</data:name>
        <data:importance>8.359547712811907E-4</data:importance>
      </row>
      <row>
        <data:name>Marital=Divorced</data:name>
        <data:importance>8.777525098452488E-4</data:importance>
      </row>
      <row>
        <data:name>Marital=Married</data:name>
        <data:importance>0.20155442366598394</data:importance>
      </row>
      <row>
        <data:name>Marital=Married-spouse-absent</data:name>
        <data:importance>0.0</data:importance>
      </row>
      <row>
        <data:name>Marital=Unmarried</data:name>
        <data:importance>0.0025007194012685165</data:importance>
      </row>
      <row>
        <data:name>Marital=Widowed</data:name>
        <data:importance>0.0</data:importance>
      </row>
      <!-- Omitted PMML content -->
    </InlineTable>
  </Extension>
  <MiningField name="Adjusted" usageType="target"/>
  <MiningField name="Education" importance="0.1063161558038196"/>
  <MiningField name="Employment" importance="0.04921320368045852"/>
  <MiningField name="Gender" importance="0.013711896190992362"/>
  <MiningField name="Marital" importance="0.20576885034837888"/>
  <MiningField name="Occupation" importance="0.13993815739579937"/>
  <MiningField name="Age" importance="0.10989324680508326"/>
  <MiningField name="Hours" importance="0.1529234886718298"/>
  <MiningField name="Income" importance="0.22223500110363806"/>
</MiningSchema>
```

The quick statistics shows that 13 out of 49 features have zero importance, which means that they are redundant from the current model perspective.
Of the remaining 36 features, the most important one is the "Marital=Married" binary indicator feature that alone does over 20% of work. Interestingly enough, all the other "Marital" category levels contribute very little.
This suggest that the "Marital" column should be encoded using some binarizing transformer instead ("Marital equals Married" vs. "Marital does not equal Married").

On aggregate, the importance of the "Marital" column is only surpassed by the "Income" column.
Its importance is obtained by summing the "Income" direct feature importance and half of the "Hourly_Income" derived feature importance (`0.14371484745257854 + 1/2 * 0.15704030730211904 = 0.22223500110363806`).

The overall ranking of columns is "Income" > "Marital" > "Hours" > "Occupation" > "Age" > "Education" > "Employment" > "Gender".
Numeric columns tend to precede string columns.
This may be caused by the fact that Scikit-Learn decision tree algorithms do underperform when categorical features have been one-hot encoded.

PMML documents are text based and very well structured, which allows for efficient information retrieval using command-line tools.

Using the [Xidel](https://github.com/benibela/xidel) tool to extract "Occupation" field importances for the "RandomForestAudit" case:

```
$ xidel --xpath "//MiningField[@name = 'Occupation']/@importance" RandomForestAudit.pmml
```

The console print-out shows 32 values.
The first value corresponds to the root model (`/PMML/MiningModel`), and the following 31 values to member decision tree models (`/PMML/MiningModel/Segmentation/Segment/TreeModel`).
All field importance values are roughly of the same magnitude.

Using the `grep` tool to extract "Occupation" field importances for the "GradientBoostingAudit" case:

```
$ grep -Po "(?<=<MiningField name=\"Occupation\" importance=\")[^\"]*(?=\"/>)" GradientBoostingAudit.pmml
```

The console print-out shows only 24 values this time.
The first value `0.1445742412830531` corresponds to the root model. The following 23 values range from `0.00341113616139356` to `0.4084976807753639`, and correspond to member decision tree models; the "missing" eight field importances should be interpreted as `0.0` values.

## Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2021-07-11/train.py" | absolute_url }})