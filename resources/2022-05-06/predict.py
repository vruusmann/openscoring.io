from jpmml_evaluator import make_evaluator

import pandas

df = pandas.read_csv("audit.csv")

evaluator = make_evaluator("DecisionTreeAudit.pmml") \
	.verify()

yt = evaluator.evaluateAll(df)
yt.to_csv("DecisionTreeAudit.csv", index = False)