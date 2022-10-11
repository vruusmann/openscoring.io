from jpmml_evaluator import make_evaluator
from jpmml_evaluator.py4j import launch_gateway, Py4JBackend

gateway = launch_gateway()

backend = Py4JBackend(gateway)

evaluator = make_evaluator(backend, "XGBoostAudit-reporting.pmml", reporting = True) \
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

gateway.shutdown()
