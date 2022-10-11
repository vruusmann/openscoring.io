from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

df_X = df[cat_columns + cont_columns]
df_y = df["Adjusted"]

mapper = DataFrameMapper(
	[(cat_column, [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
	[([cont_column], [ContinuousDomain(), StandardScaler()]) for cont_column in cont_columns]
)

classifier = LogisticRegression(multi_class = "ovr", penalty = "elasticnet", solver = "saga", max_iter = 1000)

param_grid = {
	"l1_ratio" : [0.7, 0.8, 0.9]
}

searcher = GridSearchCV(estimator = classifier, param_grid = param_grid)

pipeline = PMMLPipeline([
	("mapper", mapper),
	("searcher", searcher)
])
pipeline.fit(df_X, df_y)

print(searcher.best_params_)

pipeline.verify(df_X.sample(n = 5))

sklearn2pmml(pipeline, "GridSearchAudit.pmml")