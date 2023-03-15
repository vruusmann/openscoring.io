from jpmml_evaluator import make_evaluator

evaluator = make_evaluator("XGBoostAudit-reporting.pmml", reporting = True) \
	.verify()

arguments = {
	"Age" : 38,
	"Employment" : "Private",
	"Education" : "College",
	"Marital" : "Unmarried",
	"Occupation" : "Service",
	"Income" : 81838,
	"Gender" : "Female",
	"Deductions" : False,
	"Hours" : 72
}
print(arguments)

results = evaluator.evaluate(arguments)
print(results)
