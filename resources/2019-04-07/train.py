from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import PMMLLabelEncoder
from sklearn2pmml.pipeline import PMMLPipeline

import pandas
import warnings

# See https://stackoverflow.com/a/14463362
warnings.filterwarnings("ignore")

def train_convert(name, audit_df, mapper, feature_name = "auto", categorical_feature = "auto"):
	print("Experiment: {0}".format(name))

	lgbm_config = {
		"n_estimators" : 31,
		"objective" : "binary",
		"random_state" : 42
	}
	fit_params = {
		"classifier__feature_name" : feature_name,
		"classifier__categorical_feature" : categorical_feature
	}
	
	classifier = LGBMClassifier(**lgbm_config)
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", classifier)
	])
	pipeline.fit(audit_df, audit_df["Adjusted"], **fit_params)

	y_true = audit_df["Adjusted"]
	y_pred = pipeline.predict(audit_df)
	y_proba = pipeline.predict_proba(audit_df)
	print("\tAccuracy score: {}".format(accuracy_score(y_true, y_pred)))
	print("\tPrecision score: {}".format(precision_score(y_true, y_pred)))
	print("\tRecall score: {}".format(recall_score(y_true, y_pred)))
	print("\tROC AUC score: {}".format(roc_auc_score(y_true, y_proba[:, 1])))

	pipeline.configure(compact = False)

	sklearn2pmml(pipeline, name + ".pmml")

print("Dataset: {0}".format("Audit"))

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

train_convert("LabelBinarizer", df, mapper)

Xt = mapper.fit_transform(df)
cat_indices = [i for i in range(0, Xt.shape[1] - len(cont_columns))]

train_convert("LabelBinarizerCat", df, mapper, categorical_feature = cat_indices)

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), LabelEncoder()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

Xt = mapper.fit_transform(df)
cat_indices = [i for i in range(0, len(cat_columns))]

train_convert("LabelEncoder", df, mapper, categorical_feature = cat_indices)

print("Dataset: {0}".format("Audit-with-missing-values"))

df = pandas.read_csv("audit-NA.csv", na_values = ["N/A"])

# XXX
#train_convert("LabelEncoder", df, mapper, categorical_feature = cat_indices)

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), SimpleImputer(strategy = "most_frequent"), LabelEncoder()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

train_convert("LabelEncoderImp", df, mapper, categorical_feature = cat_indices)

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), PMMLLabelEncoder()]) for cat_column in cat_columns] +
	[(cont_columns, ContinuousDomain())]
)

train_convert("PMMLLabelEncoder", df, mapper, categorical_feature = cat_indices)