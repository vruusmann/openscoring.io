from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.ensemble import SelectFirstClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklego.meta import EstimatorTransformer
from sklego.preprocessing import IdentityTransformer

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

audit_X = df[cat_columns + cont_columns]
audit_y = df["Adjusted"]

#
# Data pre-processing
#

transformer = ColumnTransformer([
	("cont", "passthrough", cont_columns),
	# Use dense encoding for improved Scikit-Lego compatibility
	("cat", OneHotEncoder(sparse = False), cat_columns)
])

#
# Data enrichment with the anomaly score
#

outlier_detector = IsolationForest(random_state = 13)

enricher = FeatureUnion([
	("identity", IdentityTransformer()),
	#("outlier_detector", make_pipeline(EstimatorTransformer(outlier_detector, predict_func = "predict"), OneHotEncoder()))
	("outlier_detector", EstimatorTransformer(outlier_detector, predict_func = "decision_function"))
])

#
# Anomaly score-aware classification
#

def make_column_dropper(drop_cols):
	return ColumnTransformer([
		("drop", "drop", drop_cols)
	], remainder = "passthrough")

classifier = SelectFirstClassifier([
	("outlier", LinearSVC(), "X[-1] <= 0"),
	("inlier", make_pipeline(make_column_dropper([-1]), LogisticRegression()), str(True))
])

pipeline = PMMLPipeline([
	("transformer", transformer),
	("enricher", enricher),
	("classifier", classifier)
])
pipeline.fit(audit_X, audit_y)

sklearn2pmml(pipeline, "SelectFirstAudit.pmml")