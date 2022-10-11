from sklearn.experimental import enable_hist_gradient_boosting # noqa

from lightgbm import LGBMClassifier
from pandas import Series
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import PMMLLabelBinarizer
from sklearn2pmml.preprocessing.lightgbm import make_lightgbm_column_transformer
from sklearn2pmml.preprocessing.xgboost import make_xgboost_column_transformer
from xgboost import XGBClassifier

import numpy
import pandas

df = pandas.read_csv("audit-NA.csv", na_values = ["N/A", "NA"])

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

df_X = df[cat_columns + cont_columns]
df_y = df["Adjusted"]

dtypes = df_X.dtypes

mapper = ColumnTransformer(
	[(cat_column, CategoricalDomain(), [cat_column]) for cat_column in cat_columns] +
	[(cont_column, ContinuousDomain(), [cont_column]) for cont_column in cont_columns]
)

dtypes = Series(dtypes.values, index = [0, 1, 2, 3, 4, 5, 6])

lightgbm_mapper, lightgbm_categorical_feature = make_lightgbm_column_transformer(dtypes, missing_value_aware = True)
lightgbm_pipeline = Pipeline([
	("mapper", lightgbm_mapper),
	("classifier", LGBMClassifier(n_estimators = 31, max_depth = 3, random_state = 13, categorical_feature = lightgbm_categorical_feature))
])

xgboost_mapper = make_xgboost_column_transformer(dtypes, missing_value_aware = True)
xgboost_pipeline = Pipeline([
	("mapper", xgboost_mapper),
	("classifier", XGBClassifier(n_estimators = 31, max_depth = 3, random_state = 13))
])

sklearn_mapper = ColumnTransformer(
	[(str(cat_index), PMMLLabelBinarizer(sparse_output = False), [cat_index]) for cat_index in range(0, len(cat_columns))] +
	[(str(cont_index), "passthrough", [cont_index]) for cont_index in range(len(cat_columns), len(cat_columns + cont_columns))]
, remainder = "drop")
sklearn_pipeline = Pipeline([
	("mapper", sklearn_mapper),
	("classifier", HistGradientBoostingClassifier(max_iter = 31, max_depth = 3, random_state = 13))
])

final_estimator = LogisticRegression(multi_class = "ovr", random_state = 13)

# See https://stackoverflow.com/a/55326439
class DisabledCV:

	def __init__(self):
		self.n_splits = 1

	def split(self, X, y, groups = None):
		yield (numpy.arange(len(X)), numpy.arange(len(y)))

	def get_n_splits(self, X, y, groups=None):
		return self.n_splits

pipeline = PMMLPipeline([
	("mapper", mapper),
	("ensemble", StackingClassifier([
		("lightgbm", lightgbm_pipeline),
		("xgboost", xgboost_pipeline),
		("sklearn", sklearn_pipeline)
	], final_estimator = final_estimator, cv = DisabledCV(), passthrough = False))
])
pipeline.fit(df_X, df_y)
pipeline.verify(df_X.sample(n = 10, random_state = 13))

sklearn2pmml(pipeline, "StackingEnsembleAudit.pmml")