from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

import joblib
import pandas

df = pandas.read_csv("audit.csv")
print(df.dtypes)

cat_columns = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

audit_X = df[cat_columns + cont_columns]
audit_y = df["Adjusted"]

def make_fit_pmml_pipeline(classifier):
	transformer = ColumnTransformer([
		("cont", StandardScaler() if isinstance(classifier, LogisticRegression) else "passthrough", cont_columns),
		#("cat", OneHotEncoder(drop = "first" if isinstance(classifier, LogisticRegression) else None), cat_columns)
		("cat", OneHotEncoder(), cat_columns)
	])
	pipeline = PMMLPipeline([
		("transformer", transformer),
		("classifier", classifier)
	])
	pipeline.fit(audit_X, audit_y)
	return pipeline

def dump_pipeline(pipeline, name):
	joblib.dump(pipeline, name + ".pkl")
	sklearn2pmml(pipeline, name + ".pmml")

lr_pipeline = make_fit_pmml_pipeline(LogisticRegression())
dump_pipeline(lr_pipeline, "LogisticRegressionAudit")

rf_pipeline = make_fit_pmml_pipeline(RandomForestClassifier(n_estimators = 71, min_samples_split = 10, random_state = 13))
rf_pipeline.configure(compact = False, flat = False)
dump_pipeline(rf_pipeline, "RandomForestAudit")