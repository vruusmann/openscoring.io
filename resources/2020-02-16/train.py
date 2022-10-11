from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import RFormula, SQLTransformer
from pyspark2pmml import PMMLBuilder

def fit_convert_model(color):
	df = spark.read.option("delimiter", ";").csv("winequality-{}.csv".format(color), header = True, inferSchema = True)
	
	statement = """
		SELECT *,
		(`free sulfur dioxide` / `total sulfur dioxide`) AS `ratio of free sulfur dioxide`
		FROM __THIS__
	"""
	sqlTransformer = SQLTransformer(statement = statement)
	formula = "quality ~ ."
	rFormula = RFormula(formula = formula)
	classifier = DecisionTreeClassifier(minInstancesPerNode = 20)
	pipeline = Pipeline(stages = [sqlTransformer, rFormula, classifier])
	pipelineModel = pipeline.fit(df)
	
	PMMLBuilder(sc, df, pipelineModel) \
		.putOption(classifier, "compact", False) \
		.putOption(classifier, "keep_predictionCol", False) \
		.verify(df.sample(False, 0.005).limit(5)) \
		.buildFile("{}WineQuality.pmml".format(color.capitalize()))

fit_convert_model("red")
fit_convert_model("white")
