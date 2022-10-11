#
# Training
#

from lightgbm import LGBMClassifier
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing.lightgbm import make_lightgbm_dataframe_mapper

import pandas

df = pandas.read_csv("audit-NA.csv", na_values = ["N/A", "NA"])

columns = df.columns.tolist()

df_X = df[columns[: -1]]
df_y = df[columns[-1]]

# Drop boolean features
df_X = df_X.drop(["Deductions"], axis = 1)

mapper, categorical_feature = make_lightgbm_dataframe_mapper(df_X.dtypes, missing_value_aware = True)
classifier = LGBMClassifier(random_state = 13)

pipeline = PMMLPipeline([
	("mapper", mapper),
	("classifier", classifier)
])
pipeline.fit(df_X, df_y, classifier__categorical_feature = categorical_feature)

#
# Conversion
#

from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "LightGBMAudit.pmml")