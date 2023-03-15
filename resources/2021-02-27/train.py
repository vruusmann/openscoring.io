from jpmml_evaluator import make_evaluator
from mlxtend.preprocessing import DenseTransformer
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from xgboost import XGBClassifier

import pandas

df = pandas.read_csv("sentiment.csv")

X = df["Sentence"]
y = df["Score"]

def jpmml_convert_evaluate(pipeline):
	pmml_pipeline = make_pmml_pipeline(pipeline, active_fields = ["Sentence"], target_fields = ["Score"])
	sklearn2pmml(pmml_pipeline, "XGBSentiment.pmml")

	evaluator = make_evaluator("XGBSentiment.pmml") \
		.verify()

	print("jpmml: {}".format(evaluator.evaluate({"Sentence" : X[0]})))

#
# Sparse (default) pipeline
#

pipeline = Pipeline([
  ("countvectorizer", CountVectorizer()),
  ("classifier", XGBClassifier(random_state = 13))
])
pipeline.fit(X, y)

print("sparse (default): {}".format(pipeline.predict_proba(X.head(1))))

jpmml_convert_evaluate(pipeline)

#
# Dense pipeline
#

pipeline = Pipeline([
  ("countvectorizer", CountVectorizer()),
  ("densifier", DenseTransformer()),
  ("classifier", XGBClassifier(random_state = 13))
])
pipeline.fit(X, y)

print("dense: {}".format(pipeline.predict_proba(X.head(1))))

jpmml_convert_evaluate(pipeline)

#
# Sparse (zero count-aware) pipeline
#

countvectorizer = CountVectorizer()
Xt = countvectorizer.fit_transform(X)

# Convert from csr_matrix to sparse DataFrame
Xt_df_sparse = DataFrame.sparse.from_spmatrix(Xt)
print("DF density: {}".format(Xt_df_sparse.sparse.density))

classifier = XGBClassifier(random_state = 13)
classifier.fit(Xt_df_sparse, y)

pipeline = make_pipeline(countvectorizer, classifier)

print("sparse (zero count-aware): {}".format(classifier.predict_proba(Xt_df_sparse.head(1))))

jpmml_convert_evaluate(pipeline)