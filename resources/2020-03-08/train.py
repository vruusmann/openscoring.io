from pandas import DataFrame
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import Alias, DateDomain, DateTimeDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml.preprocessing import DaysSinceYearTransformer, ExpressionTransformer, SecondsSinceMidnightTransformer, SecondsSinceYearTransformer

import joblib
import pandas

# Apollo Lunar Missions
# https://nssdc.gsfc.nasa.gov/planetary/lunar/apollo.html
df = DataFrame([
	["1968-12-21T12:51:00Z", None, "1968-12-27T15:51:42Z", True], # Apollo 8
	["1969-05-18T16:49:00Z", None, "1969-05-26T16:52:23Z", True], # Apollo 10
	["1969-07-16T13:32:00Z", "1969-07-20T20:17:40Z", "1969-07-24T16:50:35Z", True], # Apollo 11
	["1969-11-14T16:22:00Z", "1969-11-19T06:54:35Z", "1969-11-24T20:58:24Z", True], # Apollo 12
	["1970-04-11T19:13:00Z", None, "1970-04-17T18:07:41Z", False], # Apollo 13
	["1971-01-31T21:03:02Z", "1971-02-05T09:18:11Z", "1971-02-09T21:05:00Z", True], # Apollo 14
	["1971-07-26T13:34:00Z", "1971-07-30T22:16:29Z", "1971-08-07T20:45:53Z", True], # Apollo 15
	["1972-04-16T17:54:00Z", "1972-04-21T02:23:35Z", "1972-04-27T19:45:05Z", True], # Apollo 16
	["1972-12-07T05:33:00Z", "1972-12-11T19:54:57Z", "1972-12-19T19:24:59Z", True], # Apollo 17
], columns = ["launch", "moon landing", "return", "success"])

print(df)

def awarestr_to_naivestr(x, tzinfo):
	# Parse aware
	x = pandas.to_datetime(x)
	# Unify timezones
	x = x.dt.tz_convert(tzinfo)
	# Convert from aware to naive
	x = x.dt.tz_localize(None)
	# Format naive
	x = x.dt.strftime("%Y-%m-%dT%H:%M:%S")
	return x

tzinfo = "Europe/Tallinn"

df["launch"] = awarestr_to_naivestr(df["launch"], tzinfo)
df["moon landing"] = awarestr_to_naivestr(df["moon landing"], tzinfo).replace({"NaT" : None})
df["return"] = awarestr_to_naivestr(df["return"], tzinfo)

print(df)

def fit_convert(mapper, name):
	pipeline = PMMLPipeline([
		("mapper", mapper),
		("classifier", DecisionTreeClassifier())
	])
	pipeline.fit(df, df["success"])
	pipeline.configure(compact = False)

	sklearn2pmml(pipeline, name + ".pmml")

mapper = DataFrameMapper([
	(["launch", "return"], [DateDomain(), DaysSinceYearTransformer(year = 1970), ExpressionTransformer("X[1] - X[0]")])
])

fit_convert(mapper, "DurationInDays")

mapper = DataFrameMapper([
	(["launch", "return"], [DateTimeDomain(), SecondsSinceYearTransformer(year = 1970), ExpressionTransformer("X[1] - X[0]")])
])

fit_convert(mapper, "DurationInSeconds")

mapper = DataFrameMapper([
	(["launch"], [DateTimeDomain(), SecondsSinceMidnightTransformer(), Alias(ExpressionTransformer("numpy.floor(X[0] / 3600)"), "HourOfLaunch", prefit = True)]),
])

fit_convert(mapper, "HourOfLaunch")