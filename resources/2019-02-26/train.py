from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

import pandas

df = pandas.read_csv("audit.csv")

pipeline = PMMLPipeline([
	("mapper", DataFrameMapper(
		[(cat_column, [CategoricalDomain(with_statistics = False), LabelBinarizer()]) for cat_column in ["Education", "Employment", "Marital", "Occupation", "Gender"]] +
		[(cont_column, [ContinuousDomain(with_statistics = False)]) for cont_column in ["Age", "Income"]]
	)),
	("classifier", XGBClassifier(objective = "binary:logistic", n_estimators = 17, seed = 13))
])
pipeline.fit(df, df["Adjusted"])

sklearn2pmml(pipeline, "XGBoostAudit.pmml")