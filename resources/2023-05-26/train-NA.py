from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.sql import SparkSession

spark = SparkSession.builder \
	.getOrCreate()

audit_df = spark.read.csv("audit-NA.csv", header = True, inferSchema = True, nanValue = "N/A")
print(audit_df.dtypes)

audit_df = audit_df.replace("N/A", None)

cat_cols = ["Deductions", "Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Hours", "Income"]

labelIndexer = StringIndexer(inputCol = "Adjusted", outputCol = "idxAdjusted")
labelIndexerModel = labelIndexer.fit(audit_df)

catColumnsIndexer = StringIndexer(inputCols = cat_cols, outputCols = ["idx" + cat_col for cat_col in cat_cols], handleInvalid = "keep")
catColumnsIndexerModel = catColumnsIndexer.fit(audit_df)

audit_df = catColumnsIndexerModel.transform(audit_df)

for outputCol, labels in zip(catColumnsIndexerModel.getOutputCols(), catColumnsIndexerModel.labelsArray):
	audit_df = audit_df.replace(to_replace = float(len(labels)), value = -999, subset = [outputCol])

vectorAssembler = VectorAssembler(inputCols = catColumnsIndexerModel.getOutputCols() + cont_cols, outputCol = "featureVector", handleInvalid = "keep")

classifier = LightGBMClassifier(categoricalSlotNames = cat_cols, slotNames = cat_cols + cont_cols, objective = "binary", numIterations = 117, labelCol = labelIndexerModel.getOutputCol(), featuresCol = vectorAssembler.getOutputCol())

pipeline = Pipeline(stages = [labelIndexerModel, vectorAssembler, classifier])
pipelineModel = pipeline.fit(audit_df)
print(pipelineModel)

pipelineModel.save("LightGBMAuditNA")
