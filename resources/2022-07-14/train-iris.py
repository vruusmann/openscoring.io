from sklearn_pandas import DataFrameMapper
from sklearn.datasets import load_iris
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.tree.chaid import CHAIDClassifier

def make_passthrough_mapper(cols):
	return DataFrameMapper(
		[([col], CategoricalDomain()) for col in cols]
	)

def make_classifier(max_depth = 5):
	config = {
		"max_depth" : max_depth
	}
	return CHAIDClassifier(config = config)

iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)

pipeline = PMMLPipeline([
	("mapper", make_passthrough_mapper(iris_X.columns.values)),
	("classifier", make_classifier(max_depth = 3))
])
pipeline.fit(iris_X, iris_y)

sklearn2pmml(pipeline, "CHAIDIris.pmml")