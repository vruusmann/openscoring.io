from openscoring import Openscoring

import pandas

os = Openscoring("http://localhost:8080/openscoring")

os.deployFile("RedWineQuality", "RedWineQuality.pmml")
os.deployFile("WhiteWineQuality", "WhiteWineQuality.pmml")

dictRequest = {
	#"fixed acidity" : 7.4,
	"volatile acidity" : 0.7,
	#"citric acid" : 0,
	"residual sugar" : 1.9,
	#"chlorides" : 0.076,
	"free sulfur dioxide" : 11,
	"total sulfur dioxide" : 34,
	#"density" : 0.9978,
	"pH" : 3.51,
	"sulphates" : 0.56,
	"alcohol" : 9.4,
}

dictResponse = os.evaluate("RedWineQuality", dictRequest)
print(dictResponse)

dfRequest = pandas.read_csv("winequality-white.csv", sep = ";")

dfResponse = os.evaluateCsv("WhiteWineQuality", dfRequest)
print(dfResponse.head(5))