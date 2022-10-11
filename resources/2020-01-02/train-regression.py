from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import numpy
import pandas

df = pandas.read_csv("auto.csv")

cat_columns = ["cylinders", "model_year", "origin"]
cont_columns = ["acceleration", "displacement", "horsepower", "weight"]

df_X = df[cat_columns + cont_columns]
df_y = df["mpg"]

mapper = DataFrameMapper(
	[([cat_column], [CategoricalDomain(), OneHotEncoder()]) for cat_column in cat_columns] +
	[([cont_column], [ContinuousDomain(), StandardScaler()]) for cont_column in cont_columns]
)

estimator = StackingRegressor([
	("dt", DecisionTreeRegressor(max_depth = 7, random_state = 13)),
	("mlp", MLPRegressor(hidden_layer_sizes = (7), solver = "lbfgs", max_iter = 2000, random_state = 13))
], final_estimator = LinearRegression(), passthrough = True)

pipeline = PMMLPipeline([
	("mapper", mapper),
	("estimator", estimator)
])
pipeline.fit(df_X, df_y)
pipeline.verify(df_X.sample(n = 10, random_state = 13))

sklearn2pmml(pipeline, "StackingEnsembleAuto.pmml")