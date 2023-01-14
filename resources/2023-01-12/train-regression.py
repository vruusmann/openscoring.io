from pycaret.regression import RegressionExperiment
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pycaret import make_pmml_pipeline as pycaret_make_pmml_pipeline

import pandas

df = pandas.read_csv("auto.csv")

cat_cols = ["cylinders", "model_year", "origin"]
for cat_col in cat_cols:
	df[cat_col] = df[cat_col].astype(str)

print(df.dtypes)

exp = RegressionExperiment()
exp.setup(
	data = df, target = "mpg",
	# Model composition changes, when omitting this attribute
	categorical_features = cat_cols,
	imputation_type = None,
	encoding_method = None, max_encoding_ohe = 3,
	normalize = True, normalize_method = "robust",
	remove_multicollinearity = True, multicollinearity_threshold = 0.9
)

# Generate models
top3_models = exp.compare_models(exclude = ["catboost", "gpc", "knn"], n_select = 3)

# Select the best model from generated models
automl_model = exp.automl(optimize = "MAE")

pycaret_pipeline = exp.finalize_model(automl_model)

pmml_pipeline = pycaret_make_pmml_pipeline(pycaret_pipeline, target_fields = ["mpg"])

sklearn2pmml(pmml_pipeline, "PyCaretAuto.pmml")