from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import KBinsDiscretizer
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import CutTransformer
from sklearn2pmml.tree.chaid import CHAIDClassifier

import numpy
import pandas

def load_audit(name):
	return pandas.read_csv(name + ".csv", na_values = ["N/A", "NA"])

def make_mapper(cat_cols, cont_cols):
	return DataFrameMapper(
		[([cat_col], CategoricalDomain()) for cat_col in cat_cols] +
		[(cont_cols, [ContinuousDomain(), KBinsDiscretizer(n_bins = 5, encode = "ordinal", strategy = "quantile")])]
	)

def make_sparse_mapper(df, cat_cols, cont_cols):
	binners = dict()
	for cont_col in cont_cols:
		bins = numpy.nanquantile(df[cont_col], q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
		# Deduplicate and convert from Numpy scalar float to Python float
		bins = [float(bin) for bin in dict.fromkeys(bins)]
		labels = list(range(0, len(bins) - 1))
		binners[cont_col] = CutTransformer(bins = bins, labels = labels)

	return DataFrameMapper(
		[([cat_col], CategoricalDomain()) for cat_col in cat_cols] +
		[([cont_col], [ContinuousDomain(), binners[cont_col]]) for cont_col in cont_cols]
	)

def make_classifier(max_depth = 5):
	config = {
		"max_depth" : max_depth
	}
	return CHAIDClassifier(config = config)

def make_fit_pipeline(df, mapper, name):
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", make_classifier())
	])
	pipeline.fit(df, df[df.columns.values[-1]])
	sklearn2pmml(pipeline, name + ".pmml")

cat_cols = ["Deductions", "Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Hours", "Income"]

df_audit = load_audit("audit")

make_fit_pipeline(df_audit, make_mapper(cat_cols, cont_cols), "CHAIDAudit")

df_audit_na = load_audit("audit-NA")

make_fit_pipeline(df_audit_na, make_sparse_mapper(df_audit_na, cat_cols, cont_cols), "CHAIDAuditNA")
