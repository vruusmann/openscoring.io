from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain, MultiDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import ExpressionTransformer, LookupTransformer

import numpy
import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

df_X = df[cat_columns + cont_columns]
df_y = df["Adjusted"]

employment_mapping = {
	"Consultant" : "Private",
	"Private" : "Private",
	"PSFederal" : "Public",
	"PSLocal" : "Public",
	"PSState" : "Public",
	"SelfEmp" : "Private",
	"Volunteer" : "Other"
}

mapper = DataFrameMapper([
	(["Income"], [ContinuousDomain(), ExpressionTransformer("numpy.log(X[0])", dtype = numpy.float64)]),
	(["Employment"], [CategoricalDomain(), LookupTransformer(employment_mapping, default_value = None), OneHotEncoder(drop = "first")]),
	(["Gender", "Marital"], [MultiDomain([CategoricalDomain(), CategoricalDomain()]), OneHotEncoder(), PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)]),
	(["Age", "Hours"], [ContinuousDomain(), StandardScaler()]), 
	(["Education"], [CategoricalDomain(), OneHotEncoder(drop = "first")]),
	(["Occupation"], [CategoricalDomain(), OneHotEncoder(drop = "first")])
])

pipeline = PMMLPipeline([
	("mapper", mapper),
	("classifier", LogisticRegression(multi_class = "ovr", max_iter = 1000))
])
pipeline.fit(df_X, df_y)

pipeline.verify(df_X.sample(n = 10))

sklearn2pmml(pipeline, "SkLearnAudit.pmml")