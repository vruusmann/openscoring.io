from jpmml_evaluator import make_evaluator
from jpmml_evaluator.pyjnius import jnius_configure_classpath, PyJNIusBackend

import pandas

jnius_configure_classpath()

df = pandas.read_csv("audit.csv")

backend = PyJNIusBackend()

evaluator = make_evaluator(backend, "DecisionTreeAudit.pmml") \
	.verify()

yt = evaluator.evaluateAll(df)
yt.to_csv("DecisionTreeAudit.csv", index = False)