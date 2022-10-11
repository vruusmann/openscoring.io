from jpmml_evaluator import make_evaluator
from jpmml_evaluator.pyjnius import jnius_configure_classpath, PyJNIusBackend

import numpy
import pandas

model_path = "CHAIDAuditNA.pmml"

input_path = "audit-NA.csv"
output_path = "audit-NA-results.csv"

jnius_configure_classpath()

backend = PyJNIusBackend()

evaluator = make_evaluator(backend, model_path) \
	.verify()

df = pandas.read_csv(input_path, na_values = ["N/A", "NA"])
print(df.dtypes)
#print(df)

# Replace float(NaN) with None in order to avoid InvalidResultException like 
# "Field <name> cannot accept user input value NaN"
df = df.replace({numpy.nan: None})

df_pred = evaluator.evaluateAll(df)
print(df_pred.dtypes)
#print(df_pred)

df_pred.to_csv(output_path, index = False)