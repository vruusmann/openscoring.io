from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

from util import *

import joblib

cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Hours", "Income"]

def slice_df(df):
	X = df[cat_cols + cont_cols]
	y = df["Adjusted"]
	return (X, y)

def make_pmml_passthrough_transformer(cat_cols, cont_cols):
	return DataFrameMapper(
		[([cont_col], [ContinuousDomain()]) for cont_col in cont_cols] +
		[([cat_col], [CategoricalDomain()]) for cat_col in cat_cols]
	, input_df = True, df_out = True)

def make_classifier(enable_categorical = False):
	return XGBClassifier(objective = "binary:logistic", tree_method = "gpu_hist", n_estimators = 71, enable_categorical = enable_categorical, random_state = 13)

def make_fit_pipeline(df, transformer, classifier):
	X, y = df
	pipeline = PMMLPipeline([
		("transformer", transformer),
		("classifier", classifier)
	])
	pipeline.fit(X, y)
	return pipeline

def make_fit_dump_pipeline(df, transformer, classifier, name):
	X, y = df
	pipeline = make_fit_pipeline(df, transformer, classifier)
	joblib.dump(pipeline, name + ".pkl")
	adjusted = DataFrame(pipeline.predict(X), columns = ["Adjusted"])
	adjusted_proba = DataFrame(pipeline.predict_proba(X), columns = ["probability(0)", "probability(1)"])
	adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	adjusted.to_csv(name + ".csv", index = False)

#
# Dense dataset
#

df = load_audit(cat_cols, cont_cols)
print(df.dtypes)

df = slice_df(df)

make_fit_dump_pipeline(df, make_dense_legacy_transformer(cat_cols, cont_cols), make_classifier(), "XGBAudit")
make_fit_dump_pipeline(df, make_pmml_passthrough_transformer(cat_cols, cont_cols), make_classifier(enable_categorical = True), "XGBAuditCat")

#
# Sparse dataset
#

df = load_audit_na(cat_cols, cont_cols)
print(df.dtypes)

df = slice_df(df)

make_fit_dump_pipeline(df, make_sparse_legacy_transformer(cat_cols, cont_cols), make_classifier(), "XGBAuditNA")
make_fit_dump_pipeline(df, make_pmml_passthrough_transformer(cat_cols, cont_cols), make_classifier(enable_categorical = True), "XGBAuditCatNA")
