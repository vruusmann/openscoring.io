import pyspark

print("PySpark version: {}".format(pyspark.__version__))

from pyspark.sql import SparkSession

spark = SparkSession.builder \
	.getOrCreate()

sc = spark.sparkContext
print("Spark version: {}".format(sc.version))

if hasattr(sc, "listFiles"):
	synapsemlResourceFiles = [scFile for scFile in sc.listFiles if "synapseml" in scFile]
	print("Spark SynapseML resource files: {}".format(synapsemlResourceFiles))

import synapse.ml.lightgbm as sml_lightgbm

print("SynapseML version: {}".format(sml_lightgbm.__version__))

from synapse.ml.lightgbm import LightGBMClassifier

classifier = LightGBMClassifier()
print(classifier)
