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

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())] +
	[(["Income", "Hours"], Alias(ExpressionTransformer("X[0] / (X[1] * 52.0)"), "Hourly_Income", prefit = True))]
)

feature_eng_pipeline = Pipeline([
	("mapper", mapper)
])

Xt = feature_eng_pipeline.fit_transform(X)
Xt = Xt.astype(float)

from sklearn2pmml import make_tpot_pmml_config
from tpot.config import classifier_config_dict

# Classes supported by TPOT
tpot_config = classifier_config_dict

# Union between classes supported by TPOT and SkLearn2PMML
tpot_pmml_config = make_tpot_pmml_config(tpot_config)

# Exclude ensemble model types
tpot_pmml_config = { key: value for key, value in tpot_pmml_config.items() if not (key.startswith("sklearn.ensemble.") or key.startswith("xgboost.")) }

# Exclude some more undesirable elementary model types
del tpot_pmml_config["sklearn.neighbors.KNeighborsClassifier"]

from tpot import TPOTClassifier

classifier = TPOTClassifier(generations = 7, population_size = 11, scoring = "roc_auc", config_dict = tpot_pmml_config, random_state = 13, verbosity = 2)
classifier.fit(Xt, y)

tpot_pipeline = classifier.fitted_pipeline_

from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

# Combine fitted sub-pipelines to a fitted pipeline
pipeline = Pipeline(feature_eng_pipeline.steps + tpot_pipeline.steps)

pmml_pipeline = make_pmml_pipeline(pipeline, active_fields = X.columns.values, target_fields = [y.name])
#pmml_pipeline.verify(X.sample(50, random_state = 13, replace = False), precision = 1e-11, zeroThreshold = 1e-11)

sklearn2pmml(pmml_pipeline, "TPOTAudit.pmml", with_repr = True)