from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import load_class_mapping, make_class_mapping_jar, sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.util import fqn

import inspect

iris_X, iris_y = load_iris(return_X_y = True, as_frame = True)

class MultinomialClassifier(LogisticRegression):

	def __init__(self, **params):
		super(MultinomialClassifier, self).__init__(penalty = None, multi_class = "multinomial", **params)

classifier = MultinomialClassifier(solver = "lbfgs")

pipeline = PMMLPipeline([
	("classifier", classifier)
])
pipeline.fit(iris_X, iris_y)

# Raises RuntimeError
#sklearn2pmml(pipeline, "MultinomialIris.pmml")

mapping = load_class_mapping()
print(len(mapping))

print(fqn(LogisticRegression))
print(fqn(MultinomialClassifier))

assert "sklearn.linear_model.LogisticRegression" not in mapping, "Has no mapping for LR public name"
assert fqn(LogisticRegression) in mapping, "Has mapping for LR fully-qualified name"
assert fqn(MultinomialClassifier) not in mapping, "Has no mapping for MC fully-qualified name"

def make_class_mapping(cls, templateCls = None):

	if not templateCls:
		templateCls = inspect.getmro(cls)[1]

	pyClazzName = fqn(cls)
	javaConverterClazzName = mapping[fqn(templateCls)]

	return {
		pyClazzName : javaConverterClazzName
	}

mapping_cust = make_class_mapping(MultinomialClassifier)
print(mapping_cust)

make_class_mapping_jar(mapping_cust, "multinomial.jar")

mapping = load_class_mapping(user_classpath = ["multinomial.jar"])

assert fqn(MultinomialClassifier) in mapping, "Has mapping for MC fully-qualified name"

sklearn2pmml(pipeline, "MultinomialIris.pmml", user_classpath = ["multinomial.jar"])
