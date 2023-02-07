from pandas import Series
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost.compat import XGBoostLabelEncoder
from xgboost.sklearn import XGBClassifier

import numpy
import pandas
import xgboost

print(xgboost.__version__)

df = pandas.read_csv("audit.csv")
df["Adjusted"] = df["Adjusted"].apply(lambda x: ("yes" if x else "no"))
print(df.dtypes)

cat_cols = ["Deductions", "Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Income", "Hours"]

X = df[cat_cols + cont_cols]
y = df["Adjusted"]

def make_mapper():
	return DataFrameMapper(
		[([cont_col], [ContinuousDomain()]) for cont_col in cont_cols] +
		[([cat_col], [CategoricalDomain(), OneHotEncoder(sparse_output = False, dtype = numpy.int8)]) for cat_col in cat_cols]
	, input_df = True, df_out = True)

def make_classifier():
	return XGBClassifier(objective = "binary:logistic", n_estimators = 131, max_depth = 6, use_label_encoder = False)

mapper = make_mapper()
classifier = make_classifier()

xgb_le = XGBoostLabelEncoder()
y = Series(xgb_le.fit_transform(y), name = "Adjusted")

classifier._le = xgb_le

pipeline = PMMLPipeline([
	("mapper", mapper),
	("classifier", classifier)
])
pipeline.fit(X, y)

classifier.save_model("Booster.bin")
classifier.save_model("Booster.json")

pipeline.configure(compact = False)

sklearn2pmml(pipeline, "XGBoostAudit-OHE.pmml")