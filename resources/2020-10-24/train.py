from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

import pandas

df = pandas.read_csv("audit.csv")

cat_columns = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_columns = ["Age", "Hours", "Income"]

mapper = DataFrameMapper(
  [([cat_column], [CategoricalDomain(), OneHotEncoder()]) for cat_column in cat_columns] +
  [([cont_column], [ContinuousDomain()]) for cont_column in cont_columns]
)
sampler = RandomOverSampler(sampling_strategy = {0 : 2000, 1 : 1000})
classifier = DecisionTreeClassifier(max_depth = 2, min_samples_leaf = 300)

imblearn_pipeline = Pipeline([
  ("mapper", mapper),
  ("sampler", sampler),
  ("classifier", classifier)
])

pmml_pipeline = PMMLPipeline([
  ("pipeline", imblearn_pipeline)
])
pmml_pipeline.fit(df, df["Adjusted"])
#classifier.pmml_feature_importances_ = classifier.feature_importances_
pmml_pipeline.configure(compact = False)
pmml_pipeline.verify(df.sample(frac = 0.01))

sklearn2pmml(pmml_pipeline, "ImbLearnAudit.pmml", with_repr = True)