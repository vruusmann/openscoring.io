---
layout: post
title: "Converting Scikit-Learn based TPOT automated machine learning (AutoML) pipelines to PMML documents"
author: vruusmann
keywords: scikit-learn tpot sklearn2pmml data-categorical
---

[TPOT](http://epistasislab.github.io/tpot/) is a tool that builds classification and regression models using genetic programming.

The main promise of AutoML is to eliminate data scientist from the ML/AI loop. 
An AutoML tool loads a dataset, and then assembles and evaluates a large number of pipelines trying to locate the global optimum.
The better the algorithm and the longer its running time, the higher the likelihood that it will come up with a model that compares favourably (at least in statistical terms) to human creation.

An AutoML tool assembles candidate pipelines from scratch, using whatever building blocks the underlying ML framework and library collection provides.
In Scikit-Learn, they are feature transformers, feature selectors and estimators.
The algorithm can vary the structure and composition of pipelines, and the parameterization of individual pipeline steps.
This puts AutoML algorithms into a league above conventional hyperparameter tuning algorithms (eg. `sklearn.model_selection.(GridSearchCV, RandomizedSearchCV)`), which can only vary the latter.

Upon success, the AutoML tool returns one or more fitted pipelines.
Such machine-generated pipelines are identical to human-generated pipelines in all technical and functional aspects. They can be converted to the PMML representation using the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package.

TPOT is made available for data scientists as a Scikit-Learn estimator, which can be fitted and used for prediction using `fit(X, y)` and `predict(X)` methods as usual.
For more sophisticated application scenarios, the fitted pipeline can be accessed directly as the `fitted_pipeline_` attribute, or converted to Python application code using the `export(path)` method.

Fitted [TPOT estimators cannot be pickled](https://github.com/EpistasisLab/tpot/issues/520) by design.
This poses a serious problem for the `sklearn2pmml` package, which operates on Pickle files rather than on in-memory Python objects.

For example, attempting to fit and convert an estimator-only `PMMLPipeline` object:

``` python
from sklearn.datasets import load_iris
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from tpot import TPOTClassifier

iris = load_iris()

pmml_pipeline = PMMLPipeline([
  ("classifier", TPOTClassifier(generations = 3, population_size = 11, random_state = 13, verbosity = 2))
])
pmml_pipeline.fit(iris.data, iris.target)

sklearn2pmml(pmml_pipeline, "TPOTIris.pmml", with_repr = True)
```

This attempt fails with a pickling error inside the `sklearn2pmml.sklearn2pmml(Pipeline: pipeline, str: pmml_output_path)` utility function:

```
Generation 1 - Current best internal CV score: 0.9733333333333334               
Generation 2 - Current best internal CV score: 0.9733333333333334               
Generation 3 - Current best internal CV score: 0.9800000000000001               
                                                                                
Best pipeline: LogisticRegression(LogisticRegression(input_matrix, C=5.0, dual=True, penalty=l2), C=5.0, dual=False, penalty=l1)
Traceback (most recent call last):
  File "main.py", line 13, in <module>
    sklearn2pmml(pipeline, "TPOTIris.pmml", with_repr = True)
  File "/usr/local/lib/python3.7/site-packages/sklearn2pmml/__init__.py", line 230, in sklearn2pmml
    pipeline_pkl = _dump(pipeline, "pipeline")
  File "/usr/local/lib/python3.7/site-packages/sklearn2pmml/__init__.py", line 176, in _dump
    joblib.dump(obj, path, compress = 3)
  File "/usr/local/lib/python3.7/site-packages/joblib/numpy_pickle.py", line 499, in dump
    NumpyPickler(f, protocol=protocol).dump(value)
  File "/usr/local/lib/python3.7/pickle.py", line 437, in dump
    self.save(obj)
  File "/usr/local/lib/python3.7/site-packages/joblib/numpy_pickle.py", line 292, in save
    return Pickler.save(self, obj)
  File "/usr/local/lib/python3.7/pickle.py", line 549, in save
    self.save_reduce(obj=obj, *rv)
  File "/usr/local/lib/python3.7/pickle.py", line 662, in save_reduce
    save(state)
  File "/usr/local/lib/python3.7/site-packages/joblib/numpy_pickle.py", line 292, in save
    return Pickler.save(self, obj)
  (clipped)
  File "/usr/local/lib/python3.7/pickle.py", line 504, in save
    f(self, obj) # Call unbound method with explicit self
  File "/usr/local/lib/python3.7/pickle.py", line 1013, in save_type
    return self.save_global(obj)
  File "/usr/local/lib/python3.7/pickle.py", line 957, in save_global
    (obj, module_name, name)) from None
_pickle.PicklingError: Can't pickle <class 'tpot.operator_utils.LogisticRegression__C'>: it's not found as tpot.operator_utils.LogisticRegression__C
```

The workaround is to fit `TPOTClassifier` in standalone mode, and create a `PMMLPipeline` object off the `TPOTClassifier.fitted_pipeline_` attribute using the `sklearn2pmml.make_pmml_pipeline(Pipeline: pipeline)` utility function:

``` python
from sklearn.datasets import load_iris
from sklearn2pmml import sklearn2pmml, make_pmml_pipeline
from tpot import TPOTClassifier

iris = load_iris()

classifier = TPOTClassifier(generations = 3, population_size = 11, random_state = 13, verbosity = 2)
classifier.fit(iris.data, iris.target)

pmml_pipeline = make_pmml_pipeline(classifier.fitted_pipeline_, active_fields = iris.feature_names, target_fields = ["species"])

sklearn2pmml(pmml_pipeline, "TPOTIris.pmml", with_repr = True)
```

The "iris" dataset is good for a quick demonstration that AutoML is nothing special from the PMML perspective.

The lesson is that the PMML representation is only concerned with the final state - the deployable model. The PMML representation is not concerned with the specifics of the AutoML tool/algorithm, the initial state, or any of the intermediate states of the search process.

Working with real-life datasets is only a little bit more complicated.

### Feature engineering

TPOT estimators require that the `X` argument of the `fit(X, y)` method is a numeric matrix.

If the dataset contains categorical string features, then they either need to be transformed to numeric features, or dropped.

Also, it is always advisable to enrich the dataset with custom features.
The current generation of AutoML algorithms are limited to scaling or applying unary transformations (eg. log transformation) to individual columns.
They may fall short in enumerating and trying out higher order transformations.
For example, if the domain knowledge suggests that feature ratios might be significant, then the dataset should be enhanced with derived numeric feature columns (eg. iterating over relevant numeric features, and dividing them one by one with all other relevant numeric features).

An AutoML algorithm should have no problem going through arbitrary size data matrices by applying feature selection.

The suggested approach is to split the workflow into two parts.
First, there is a feature engineering part, which accepts a raw data matrix, and transforms it to a 2-D Numpy array.
Second, there is a TPOT part, which performs the magic.

These two parts are executed one after another.
They produce fitted "child" `Pipeline` objects, which are joined programmatically into a fitted "parent" `PMMLPipeline` object for a quick and easy conversion to the PMML representation.

Sample usage:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import ExpressionTransformer

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

X = df[cat_columns + cont_columns]
y = df["Adjusted"]

feature_eng_pipeline = Pipeline([
  ("mapper", DataFrameMapper(
    [([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
    [(cont_columns, ContinuousDomain())] +
    [(["Income", "Hours"], Alias(ExpressionTransformer("X[0] / (X[1] * 52)"), "Hourly_Income", prefit = True))]
  ))
])
Xt = feature_eng_pipeline.fit_transform(X)
Xt = Xt.astype(float)

from tpot import TPOTClassifier

classifier = TPOTClassifier(generations = 7, population_size = 11, scoring = "roc_auc", random_state = 13, verbosity = 2)
classifier.fit(Xt, y)

tpot_pipeline = classifier.fitted_pipeline_

from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

pipeline = Pipeline(feature_eng_pipeline.steps + tpot_pipeline.steps)

pmml_pipeline = make_pmml_pipeline(pipeline, active_fields = X.columns.values, target_fields = [y.name])
#pmml_pipeline.verify(X.sample(50, random_state = 13, replace = False), precision = 1e-11, zeroThreshold = 1e-11)

sklearn2pmml(pmml_pipeline, "TPOTAudit.pmml", with_repr = True)
```

### Configuring TPOT search space

The [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) library (that powers the `sklearn2pmml` package) must recognize and support all pipeline steps for the conversion to succeed.

The list of [supported Scikit-Learn and third-party library transformers and estimators](https://github.com/jpmml/jpmml-sklearn/blob/master/pmml-sklearn/src/main/resources/META-INF/sklearn2pmml.properties) is long and keeps growing longer with each new release.
However, there is still a fair chance that TPOT wants to use some unsupported classes here and there.
It can be frustrating to see great compute efforts go to waste only because the JPMML-SkLearn library rejects one pipeline step out of hundreds.

TPOT estimators can be ordered to limit or expand their search space using the `config_dict` parameter.
There are two built-in config dictionaries `tpot.config.classifier_config_dict` and `tpot.config.regressor_config_dict`, which serve as defaults for classification and regression tasks, respectively.

A config dictionary can be made PMML compatible by excluding all mappings where they key is an unsupported Python class name.
The list of supported class names can be obtained by going through all the JAR files on the `sklearn2pmml` package classpath (consisting of the JPMML-SkLearn library and its third-party plugin libraries) and collecting keys from `META-INF/sklearn2pmml.properties` property files.
The whole procedure is conveniently implemented in the `sklearn2pmml.make_tpot_pmml_config(dict: config)` utility function.

``` python
from sklearn2pmml import make_tpot_pmml_config
from tpot.config import classifier_config_dict, regressor_config_dict

tpot_config = dict(classifier_config_dict)

tpot_pmml_config = make_tpot_pmml_config(tpot_config)
```

Expert users may drop further mappings from the config dictionary.
For example, AutoML algorithms tend to prefer ensemble methods for fitting.
If the goal is to obtain human-interpretable models, then it is easy to disable ensemble methods by simply dropping all mappings where the key starts with "sklearn.ensemble".

``` python
tpot_pmml_config = { key: value for key, value in tpot_pmml_config.items() if not (key.startswith("sklearn.ensemble.") or key.startswith("xgboost.")) }

del tpot_pmml_config["sklearn.neighbors.KNeighborsClassifier"]
```

### Interpreting results

TPOT defines and uses the `tpot.builtins.StackingEstimator` meta-estimator to chain multiple estimators together.

First, an estimator is fitted using the user-supplied data matrix.
This estimator is used for prediction, and its `predict(X)` and `predict_proba(X)` result columns are appended to the data matrix.
Next, another estimator is fitted using the enhanced data matrix.
If this esimator fails to improve the model (based on predefined evaluation criteria), then the search process is terminated. If it improves, its prediction is appended to data matrix, and the search process continues.

For example, the newly generated PMML document `TPOTAudit.pmml` contains a two-stage model chain, where the initial prediction by a Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`) classifier is refined by a Logistic Regression (`sklearn.linear_model.LogisticRegression`) classifier:

``` xml
<MiningModel>
  <!-- Omitted MiningSchema, ModelStats and LocalTransformations elements -->
  <Segmentation multipleModelMethod="modelChain">
    <Segment id="1">
      <True/>
      <NaiveBayesModel threshold="0.0" functionName="classification">
        <!-- Omitted MiningSchema element -->
        <Output>
          <OutputField name="probability(stack(47), 0)" optype="continuous" dataType="double" feature="probability" value="0" isFinalResult="false"/>
          <OutputField name="probability(stack(47), 1)" optype="continuous" dataType="double" feature="probability" value="1" isFinalResult="false"/>
          <OutputField name="stack(47)" optype="categorical" dataType="integer" feature="predictedValue"/>
        </Output>
        <!-- Omitted LocalTransformations, BayesInputs and BayesOutput elements -->
      </NaiveBayesModel>
    </Segment>
    <Segment id="2">
      <RegressionModel functionName="classification" normalizationMethod="logit">
        <MiningSchema>
          <MiningField name="Adjusted" usageType="target"/>
          <MiningField name="Age"/>
          <MiningField name="Education"/>
          <MiningField name="Employment"/>
          <MiningField name="Hours"/>
          <MiningField name="Income"/>
          <MiningField name="Marital"/>
          <MiningField name="Occupation"/>
          <MiningField name="probability(stack(47), 0)"/>
          <MiningField name="probability(stack(47), 1)"/>
          <MiningField name="stack(47)"/>
        </MiningSchema>
        <Output>
          <OutputField name="probability(0)" optype="continuous" dataType="double" feature="probability" value="0"/>
          <OutputField name="probability(1)" optype="continuous" dataType="double" feature="probability" value="1"/>
        </Output>
        <!-- Omitted LocalTransformations and RegressionTable elements -->
      </RegressionModel>
    </Segment>
  </Segmentation>
</MiningModel>
```

Most ML frameworks and libraries do not know or care about the origin and deeper meaning of individual columns in the training dataset.
When fitted models are converted to the PMML representation, then it becomes possible to observe all kinds of bizarre computations, starting from no-op transformations and leading to non-sensical and outright (information-) destructive ones.

For example, TPOT is casually generating model chains, where the predictions of earlier estimators are not used by any of subsequent estimators, meaning that all their computation efforts are provably wasted.

Good PMML converters such as all JPMML-family conversion libraries can run static analyses on PMML class model objects and correct many such issues.
Corrected PMML documents have lower resource requirements and perform significantly better.

### Resources

* "Audit" dataset: [`audit.csv`]({{ "/resources/data/audit.csv" | absolute_url }})
* Python script: [`train.py`]({{ "/resources/2019-06-10/train.py" | absolute_url }})
* TPOT PMML document: [`TPOTAudit.pmml`]({{ "/resources/2019-06-10/TPOTAudit.pmml" | absolute_url }})