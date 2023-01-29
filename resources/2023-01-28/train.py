from lifelines.datasets import load_lung
from pandas import CategoricalDtype, Int64Dtype
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from xgboost import Booster, DMatrix
from xgboost.sklearn import XGBRegressor

import numpy
import pandas
import xgboost

df = load_lung()

print(df.dtypes)
print(df.isna().sum())

cols = df.columns.tolist()

for col in cols:
	has_missing = pandas.isnull(df[col]).any()
	df[col] = df[col].astype(Int64Dtype() if has_missing else int)

print(df.dtypes)
print(df.isna().sum())

cat_cols = ["inst", "sex"]
cont_cols = ["age", "ph.ecog", "ph.karno", "pat.karno", "meal.cal", "wt.loss"]

def make_cat_dtype(x):
	categories = pandas.unique(x)
	# Drop null-like category levels
	categories = numpy.delete(categories, pandas.isnull(categories))
	return CategoricalDtype(categories = categories, ordered = False)

inst_dtype = make_cat_dtype(df["inst"])
sex_dtype = make_cat_dtype(df["sex"])

transformer = DataFrameMapper(
	[(["inst"], CategoricalDomain(dtype = inst_dtype))] +
	[(["sex"], CategoricalDomain(dtype = sex_dtype))] +
	[([cont_col], ContinuousDomain(dtype = numpy.float32)) for cont_col in cont_cols]
, input_df = True, df_out = True)

def make_aft_label(time, status):
	time_lower = time

	time_upper = time.copy()
	time_upper[status == 0] = float("+Inf")

	return (time_lower, time_upper)

Xt = transformer.fit_transform(df)

time_lower, time_upper = make_aft_label(df["time"], df["status"])

dmat = DMatrix(
	# Features
	data = Xt,
	# Label
	label_lower_bound = time_lower, label_upper_bound = time_upper,
	missing = float("NaN"), 
	enable_categorical = True
)

params = {
	"objective" : "survival:aft",
	"eval_metric" : "aft-nloglik",
	"max_depth" : 3,
	"tree_method" : "hist"
}

booster = xgboost.train(params = params, dtrain = dmat, num_boost_round = 31)
booster.save_model("booster.json")

booster_time = booster.predict(dmat)
print(booster_time)

regressor = XGBRegressor()
regressor._Booster = Booster(model_file = "booster.json")

pipeline = Pipeline([
	("transformer", transformer),
	("regressor", regressor)
])

pipeline_time = pipeline.predict(df)
print(pipeline_time)

def check_predict(expected, actual, rtol, atol):
	isclose = numpy.isclose(expected, actual, rtol = rtol, atol = atol, equal_nan = False)
	num_conflicts = numpy.sum(isclose == False)
	if num_conflicts:
		for idx, status in enumerate(isclose):
			if not status:
				print("{} != {}".format(expected[idx], actual[idx]))
		raise ValueError("Found {} conflicting prediction(s)".format(num_conflicts))
	print("All correct")

check_predict(booster_time, pipeline_time, 1e-6, 1e-3)

pmml_pipeline = make_pmml_pipeline(pipeline, active_fields = (cat_cols + cont_cols), target_fields = ["time"])

Xt_imp = booster.get_score(importance_type = "weight")
print(Xt_imp)

# Transform dict to list
Xt_imp = [Xt_imp[col] for col in pmml_pipeline.active_fields]

regressor.pmml_feature_importances_ = numpy.asarray(Xt_imp)

df_verif = df[pmml_pipeline.active_fields].sample(10)

pmml_pipeline.verify(df_verif, precision = 1e-6, zeroThreshold = 1e-3)

sklearn2pmml(pmml_pipeline, "XGBoostAFTLung.pmml")
