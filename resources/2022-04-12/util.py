from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import PMMLLabelBinarizer

import numpy
import pandas

def cast_cols(X, cols, dtype):
	for col in cols:
		X[col] = X[col].astype(dtype)

def load_audit(cat_cols, cont_cols):
	df = pandas.read_csv("audit.csv")
	df = df[cat_cols + cont_cols + ["Adjusted"]]
	cast_cols(df, cat_cols, "category")
	return df

def load_audit_na(cat_cols, cont_cols):
	df = pandas.read_csv("audit-NA.csv", na_values = ["N/A", "NA"])
	df = df[cat_cols + cont_cols + ["Adjusted"]]
	cast_cols(df, cat_cols, "category")
	return df

def make_dense_legacy_transformer(cat_cols, cont_cols, sparse = False, dtype = numpy.uint8):
	return ColumnTransformer([
		("cont", "passthrough", cont_cols),
		("cat", OneHotEncoder(sparse = sparse, dtype = dtype), cat_cols)
	], sparse_threshold = 1.0)

def make_sparse_legacy_transformer(cat_cols, cont_cols):
	return ColumnTransformer(
		[(cont_col, "passthrough", [cont_col]) for cont_col in cont_cols] +
		[(cat_col, PMMLLabelBinarizer(sparse_output = True), [cat_col]) for cat_col in cat_cols]
	, sparse_threshold = 1.0)