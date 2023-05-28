from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.sql import SparkSession

spark = SparkSession.builder \
	.getOrCreate()

audit_df = spark.read.csv("audit.csv", header = True, inferSchema = True)
print(audit_df.dtypes)

cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Hours", "Income"]

labelIndexer = StringIndexer(inputCol = "Adjusted", outputCol = "idxAdjusted")
labelIndexerModel = labelIndexer.fit(audit_df)

catColumnsIndexer = StringIndexer(inputCols = cat_cols, outputCols = ["idx" + cat_col for cat_col in cat_cols])

vectorAssembler = VectorAssembler(inputCols = catColumnsIndexer.getOutputCols() + cont_cols, outputCol = "featureVector")

classifier = LightGBMClassifier(objective = "binary", numIterations = 117, labelCol = labelIndexerModel.getOutputCol(), featuresCol = vectorAssembler.getOutputCol())

pipeline = Pipeline(stages = [labelIndexerModel, catColumnsIndexer, vectorAssembler, classifier])
pipelineModel = pipeline.fit(audit_df)
print(pipelineModel)

pipelineModel.save("LightGBMAudit")
