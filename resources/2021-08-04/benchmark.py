from jpmml_evaluator import make_evaluator
from jpmml_evaluator.pyjnius import jnius_configure_classpath, PyJNIusBackend
from jpmml_evaluator.py4j import launch_gateway, Py4JBackend
from pandas import DataFrame
from sklearn.pipeline import Pipeline

import joblib
import numpy
import pandas
import pickle
import sys
import time

cat_columns = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

class DummyEstimator:

	def prepare(self, X):
		data = []
		for i in range(X.shape[0]):
			entry = {"Adjusted" : "0", "probability(0)" : 1.0, "probability(1)" : 0.0}
			data.append(entry)
		self.pkl = pickle.dumps(data)

	def predict(self, X):
		pickle.dumps(X, protocol = 2)
		return pickle.loads(self.pkl)

class PMMLEstimator:

	def __init__(self, evaluator):
		self.evaluator = evaluator

	def predict(self, X):
		return self.evaluator.evaluateAll(X)

def main(estimatorFile, csvFile, flag):
	if estimatorFile.endswith(".pkl"):
		if flag == "Dummy":
			estimator = DummyEstimator()
		else:
			estimator = joblib.load(estimatorFile)
	elif estimatorFile.endswith(".pmml"):
		if flag == "JPMML/PyJNIus":
			jnius_configure_classpath()
			backend = PyJNIusBackend()
			estimator = PMMLEstimator(make_evaluator(backend, estimatorFile))
		elif flag == "JPMML/Py4J":
			gateway = launch_gateway()
			backend = Py4JBackend(gateway)
			estimator = PMMLEstimator(make_evaluator(backend, estimatorFile))
		elif flag == "PyPMML":
			from pypmml import Model
			estimator = Model.load(estimatorFile)
		else:
			raise ValueError(flag)
	else:
		raise ValueError(estimatorFile)
	#print(estimator)
	data = pandas.read_csv(csvFile)
	#print(data.dtypes)

	if isinstance(estimator, PMMLEstimator):
		sample = data.sample(n = 100000, replace = True, random_state = 13)

		sample_X = sample[cat_columns + cont_columns]

		records = sample_X.to_dict(orient = "records")
		for record in records:
			estimator.evaluator.evaluate(record)

	if isinstance(estimator, Pipeline) and flag in ["model", "transformers"]:
		transformers = Pipeline(estimator.steps[0:-1])
		model = estimator._final_estimator
		if flag == "transformers":
			estimator = transformers
		elif flag == "model":
			estimator = model
		else:
			raise ValueError()

	print("| Configuration | Time (sec) | Time per row (microsec) |")
	print("|---|---|---|")

	for config in ["1000 * 1", "1000 * 10", "1000 * 100", "10 * 1000", "10 * 10000", "1 * 100000"]:
		cycles, batch_size = (int(num) for num in config.split("*"))
		
		sample = data.sample(n = batch_size, replace = True, random_state = 13)

		sample_X = sample[cat_columns + cont_columns]
		sample_y = sample["Adjusted"]

		if hasattr(estimator, "prepare"):
			estimator.prepare(sample_X)

		if flag == "model":
			sample_X = transformers.transform(sample_X)

		num_scored = 0
		start_time = time.time()
		for cycle in range(0, cycles):
			if hasattr(estimator, "predict"):
				estimator.predict(sample_X)
			elif hasattr(estimator, "transform"):
				estimator.transform(sample_X)
			else:
				raise ValueError()
			num_scored += sample.shape[0]
		end_time = time.time()
		elapsed_time = (end_time - start_time)
		print("| {0} | {1:0.6f} | {2:0.3f} |".format(config, elapsed_time, (elapsed_time * 1000.0 * 1000.0) / num_scored))

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)