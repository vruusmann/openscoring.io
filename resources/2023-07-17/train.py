from h2o import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.model.extensions import VariableImportance
from sklearn.compose import ColumnTransformer
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.h2o import H2OFrameConstructor

import dill
import h2o
import joblib
import pandas

h2o.init()

audit_df = pandas.read_csv("audit.csv")

cat_cols = ["Deductions", "Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Hours", "Income"]

audit_X = audit_df[cat_cols + cont_cols]
audit_y = audit_df["Adjusted"]

h2o_audit_y = H2OFrame(audit_y.to_frame(), column_types = ["categorical"])

initializer = ColumnTransformer(
	[(cat_col, CategoricalDomain(), [cat_col]) for cat_col in cat_cols] +
	[(cont_col, ContinuousDomain(), [cont_col]) for cont_col in cont_cols]
)

uploader = H2OFrameConstructor()

classifier = H2ORandomForestEstimator(ntrees = 31, seed = 42)

# False before fitting
assert not isinstance(classifier, VariableImportance)

pipeline = PMMLPipeline([
	("initializer", initializer),
	("uploader", uploader),
	("classifier", classifier)
])
pipeline.fit(audit_X, h2o_audit_y)

# True after fitting
assert isinstance(classifier, VariableImportance)

with open("H2ORandomForestAudit.pkl", "wb") as pkl_file:
	#joblib.dump(pipeline, pkl_file)
	dill.dump(pipeline, pkl_file)

sklearn2pmml(pipeline, "H2ORandomForestAudit.pmml", dump_flavour = "dill")

h2o.shutdown()
