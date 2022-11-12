from h2o import H2OFrame
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn.pipeline import Pipeline

import h2o
import pandas

h2o.init()

df = pandas.read_csv("audit.csv")

X = df[[column for column in df.columns if column != "Adjusted"]]
y = df["Adjusted"]

#
# Manual data upload
#

pipeline = Pipeline([
	("classifier", H2OGeneralizedLinearEstimator())
])
# Raises AttributeError
#pipeline.fit(X, y)

h2o_X = H2OFrame(X)
h2o_y = H2OFrame(y.to_frame(), column_types = ["categorical"])

pipeline.fit(h2o_X, h2o_y)

#
# Semi-automated data upload
#

from sklearn2pmml.preprocessing.h2o import H2OFrameConstructor

pipeline = Pipeline([
	("uploader", H2OFrameConstructor()),
	("classifier", H2OGeneralizedLinearEstimator())
])
pipeline.fit(X, H2OFrame(y.to_frame(), column_types = ["categorical"]))