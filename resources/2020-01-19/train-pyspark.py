from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RFormula, SQLTransformer
from pyspark.sql.types import StringType
from pyspark2pmml import PMMLBuilder

df = spark.read.csv("audit.csv", header = True, inferSchema = True)
df = df.withColumn("Adjusted", df["Adjusted"].cast(StringType()))

statement = """
	SELECT *, 
	ln(Income) AS Log_Income,
	CASE
		WHEN Employment = "Consultant" THEN "Private"
		WHEN Employment = "Private" THEN "Private"
		WHEN Employment = "PSFederal" THEN "Public"
		WHEN Employment = "PSLocal" THEN "Public"
		WHEN Employment = "PSState" THEN "Public"
		WHEN Employment = "SelfEmp" THEN "Private"
		WHEN Employment = "Volunteer" THEN "Other"
	END AS Revalue_Employment
	FROM __THIS__
	"""
sqlTransformer = SQLTransformer(statement = statement)

formula = "Adjusted ~ . - Income - Employment + Gender:Marital"
rFormula = RFormula(formula = formula)

classifier = LogisticRegression()

pipeline = Pipeline(stages = [sqlTransformer, rFormula, classifier])
pipelineModel = pipeline.fit(df)

pmmlBuilder = PMMLBuilder(sc, df, pipelineModel) \
	.verify(df.sample(False, 0.005)) \
	.putOption(classifier, "representation", "RegressionModel")

pmmlBuilder.buildFile("PySparkAudit.pmml")
