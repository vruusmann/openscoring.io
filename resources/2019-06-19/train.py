from lightgbm import LGBMClassifier
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.ensemble import GBDTLRClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from xgboost import XGBClassifier

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

label_column = "Adjusted"

def make_fit_gbdtlr(gbdt, lr):
	mapper = DataFrameMapper(
		[([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
		[(cont_columns, ContinuousDomain())]
	)
	classifier = GBDTLRClassifier(gbdt, lr)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(df[cat_columns + cont_columns], df[label_column])
	return pipeline

pipeline = make_fit_gbdtlr(GradientBoostingClassifier(n_estimators = 499, max_depth = 2), LogisticRegression())
sklearn2pmml(pipeline, "GBDT+LR.pmml")

pipeline = make_fit_gbdtlr(RandomForestClassifier(n_estimators = 31, max_depth = 6), LogisticRegression())
sklearn2pmml(pipeline, "RF+LR.pmml")

pipeline = make_fit_gbdtlr(XGBClassifier(n_estimators = 299, max_depth = 3), LogisticRegression())
sklearn2pmml(pipeline, "XGB+LR.pmml")

def make_fit_lgbmlr(gbdt, lr):
	mapper = DataFrameMapper(
		[([cat_column], [CategoricalDomain(), LabelEncoder()]) for cat_column in cat_columns] +
		[(cont_columns, ContinuousDomain())]
	)
	classifier = GBDTLRClassifier(gbdt, lr)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(df[cat_columns + cont_columns], df[label_column], classifier__gbdt__categorical_feature = range(0, len(cat_columns)))
	return pipeline

pipeline = make_fit_lgbmlr(LGBMClassifier(n_estimators = 71, max_depth = 5), LogisticRegression())
sklearn2pmml(pipeline, "LGBM+LR.pmml")