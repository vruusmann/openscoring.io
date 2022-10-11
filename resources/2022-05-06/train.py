from category_encoders import BaseNEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import Alias
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.postprocessing import BusinessDecisionTransformer
from sklearn2pmml.preprocessing import CutTransformer, ExpressionTransformer, LookupTransformer

import numpy
import pandas

df = pandas.read_csv("audit.csv")

cat_cols = ["Education", "Employment", "Marital", "Occupation"]
cont_cols = ["Age", "Hours", "Income"]

X = df[cat_cols + cont_cols]
y = df["Adjusted"]

# XXX
#y = y.astype(bool)

mapper = ColumnTransformer([
	("cont", "passthrough", cont_cols),
	("cat", BaseNEncoder(base = 3, handle_missing = "error", handle_unknown = "error"), cat_cols)
])

classifier = DecisionTreeClassifier(min_samples_leaf = 20, random_state = 13)

pipeline = PMMLPipeline([
	("mapper", mapper),
	("classifier", classifier)
])
pipeline.fit(X, y)

#
# Class post-processing
#

binary_decisions = [
	("yes", "Auditing is needed"),
	("no", "Auditing is not needed")
]

pipeline.predict_transformer = Alias(BusinessDecisionTransformer(ExpressionTransformer("'yes' if X[0] == 1 else 'no'"), "Is auditing necessary?", binary_decisions, prefit = True), "binary decision", prefit = True)

y_predict = pipeline.predict_transform(X)
print(y_predict)

#
# Probability distribution post-processing
#

graded_decisions = [
	("no", "Auditing is not needed"),
	("no over yes", "Audit in last order"),
	("yes over no", "Audit in first order"),
	("yes", "Auditing is needed"),
]

event_proba = pipeline.predict_proba(X)[:, 1]
event_proba_quantiles = numpy.percentile(event_proba, [0, 50, 80, 95]).tolist() + [1.0]

pipeline.predict_proba_transformer = Pipeline([
	("selector", ExpressionTransformer("X[1]")),
	("decider", Alias(BusinessDecisionTransformer(CutTransformer(bins = event_proba_quantiles, labels = [key for key, value in graded_decisions]), "Is auditing necessary?", graded_decisions, prefit = True), "graded decision", prefit = True))
])

y_predict_proba = pipeline.predict_proba_transform(X)
print(y_predict_proba)

#pipeline.configure(compact = False, numeric = False)
#sklearn2pmml(pipeline, "DecisionTreeAudit.pmml")

#
# Node post-processing
#

def leaf_sizes(tree):
	leaf_sizes = dict()
	for i in range(tree.node_count):
		if (tree.children_left[i] == -1) and (tree.children_right[i] == -1):
			leaf_sizes[i] = int(tree.n_node_samples[i])
	return leaf_sizes

pipeline.apply_transformer = Alias(LookupTransformer(leaf_sizes(classifier.tree_), default_value = -1), "leaf size", prefit = True)

y_apply = pipeline.apply_transform(X)
print(y_apply)

pipeline.configure(compact = False, flat = False, numeric = False, winner_id = True)
sklearn2pmml(pipeline, "DecisionTreeAudit.pmml")