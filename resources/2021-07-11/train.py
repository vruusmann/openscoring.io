from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import Alias, CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import ExpressionTransformer
from sklearn2pmml.pipeline import PMMLPipeline

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

audit_X = df[cat_columns + cont_columns]
audit_y = df["Adjusted"]

def make_fit_pmml_pipeline(classifier, standardize = False):
	mapper = DataFrameMapper([
		(cont_columns, StandardScaler() if standardize else None),
		(["Income", "Hours"], [Alias(ExpressionTransformer("X[0] / (X[1] * 52.0)", dtype = float), "Hourly_Income", prefit = True)] + ([StandardScaler()] if standardize else [])),
		(cat_columns, OneHotEncoder())
	])
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	return pipeline

dtc = DecisionTreeClassifier(random_state = 13)
pipeline = make_fit_pmml_pipeline(dtc)
dtc.pmml_feature_importances_ = dtc.feature_importances_
sklearn2pmml(pipeline, "DecisionTreeAudit.pmml")

rfc = RandomForestClassifier(n_estimators = 31, random_state = 13)
pipeline = make_fit_pmml_pipeline(rfc)
rfc.pmml_feature_importances_ = rfc.feature_importances_
for rfc_dtc in rfc.estimators_:
	rfc_dtc.pmml_feature_importances_ = rfc_dtc.feature_importances_
sklearn2pmml(pipeline, "RandomForestAudit.pmml")

gbc = GradientBoostingClassifier(n_estimators = 31, random_state = 13)
pipeline = make_fit_pmml_pipeline(gbc)
gbc.pmml_feature_importances_ = gbc.feature_importances_
for gbc_dtr in gbc.estimators_[:, 0]:
	gbc_dtr.pmml_feature_importances_ = gbc_dtr.feature_importances_
sklearn2pmml(pipeline, "GradientBoostingAudit.pmml")

from sklearn.inspection import permutation_importance

lr = LogisticRegression()
pipeline = make_fit_pmml_pipeline(lr, standardize = True)
result = permutation_importance(pipeline, audit_X, audit_y, random_state = 13)
pipeline.pmml_feature_importances_ = result.importances_mean
sklearn2pmml(pipeline, "LogisticRegressionAudit.pmml")
