from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import SelectKBest
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
	[(cat_column, [CategoricalDomain(invalid_value_treatment = "as_is"), LabelBinarizer()]) for cat_column in cat_columns] +
	[([cont_column], [ContinuousDomain(invalid_value_treatment = "as_is"), StandardScaler()]) for cont_column in cont_columns]
)

selector = SelectKBest()

classifier = LogisticRegression(multi_class = "ovr", penalty = "elasticnet", solver = "saga", max_iter = 1000)

pipeline = PMMLPipeline([
	("mapper", mapper),
	("selector", selector),
	("classifier", classifier)
])

param_grid = {
	"selector__k" : [10, 20, 30],
	"classifier__l1_ratio" : [0.7, 0.8, 0.9]
}

searcher = GridSearchCV(estimator = pipeline, param_grid = param_grid)
searcher.fit(df_X, df_y)

print(searcher.best_params_)

best_pipeline = searcher.best_estimator_
best_pipeline.verify(df_X.sample(n = 5))

sklearn2pmml(best_pipeline, "GridSearchAudit.pmml")