from pycaret.classification import ClassificationExperiment
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pycaret import make_pmml_pipeline as pycaret_make_pmml_pipeline

import pandas

df = pandas.read_csv("audit-NA.csv")
df = df.drop(columns = ["Deductions"], axis = 1)

print(df.dtypes)

exp = ClassificationExperiment()
exp.setup(
	data = df, target = "Adjusted",
	imputation_type = "simple",
	rare_to_value = 0.02, rare_value = "(Other)",
	encoding_method = None, max_encoding_ohe = 7,
	fix_imbalance = True,
	normalize = "zscore",
	remove_multicollinearity = True, multicollinearity_threshold = 0.75
)

model = exp.create_model(estimator = "lr")

pycaret_pipeline = exp.finalize_model(model)

pmml_pipeline = pycaret_make_pmml_pipeline(pycaret_pipeline, target_fields = ["Adjusted"])

sklearn2pmml(pmml_pipeline, "PyCaretAuditNA.pmml")
