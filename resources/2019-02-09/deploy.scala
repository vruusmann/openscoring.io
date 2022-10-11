import java.io.File
import org.jpmml.evaluator.LoadingModelEvaluatorBuilder

val evaluatorBuilder = new LoadingModelEvaluatorBuilder().load(new File("LogisticRegressionAudit.pmml"))

val evaluator = evaluatorBuilder.build()

evaluator.verify()

import org.jpmml.evaluator.spark.TransformerBuilder

val transformerBuilder = new TransformerBuilder(evaluator).withTargetCols().withOutputCols().exploded(true)

val transformer = transformerBuilder.build()

var inputDs = spark.read.format("csv").option("header", "true").load("audit.csv")
inputDs = inputDs.drop("Adjusted")
inputDs.printSchema()

var resultDs = transformer.transform(inputDs)
resultDs = resultDs.select("Adjusted", "probability(0)", "probability(1)")
resultDs.printSchema()

resultDs.show(10)