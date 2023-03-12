---
layout: post
title: "Extending Scikit-Learn with date and datetime features"
author: vruusmann
keywords: scikit-learn sklearn2pmml feature-domain data-temporal
---

Scikit-Learn algorithms operate on numerical data.
If the dataset contains complex features, then they need to be explicitly encoded and/or transformed from their native high-level representation to a suitable low-level representation.
For example, a string column must be expanded into a list of binary indicator features using `LabelBinarizer` or `OneHotEncoder` transformers.

Scikit-Learn can be extended with custom features by building extension layers on top of the numeric base layer.

Custom features allow data scientists to represent and manipulate data using more realistic concepts, thereby improving their productivity (reducing cognitive load, eliminating whole categories of systematic errors).
For example, compare working with temporal data in the form of Unix timestamps (number of seconds since the Unix Epoch) versus ISO 8601 strings.

This blog post demonstrates how the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package extends Scikit-Learn with PMML-compatible date and datetime features.

## Temporal data ##

A datetime is a data structure that represents an instant (point in time) according to some calendar and time zone.

The calendar component takes care of mapping larger periods of time such as years, months and days.
Most computer systems use the [Gregorian calendar](https://en.wikipedia.org/wiki/Gregorian_calendar), which provides a rather simple algorithm for spacing 365.2422-day solar years uniformly.

The time zone component takes care of mapping periods within a day.
While based on solar time, they include socially and economically motivated adjustments.
From the software development perspective, time zones should be regarded as ever-growing lookup tables that need to be updated regularly (typically handled behind the scenes by the operating system).
The lookup function returns a time zone offset relative to the [Coordiated Universal Time (UTC)](https://en.wikipedia.org/wiki/Coordinated_Universal_Time) for the specified point in time.

Datetimes should be formatted as strings following the [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) standard:

* `<date part>` - Local date
* `<date part>T<time part>` - Local datetime
* `<date part>T<time part>±<time zone offset part>` - Local datetime with explicit time zone offset
* `<date part>T<time part>Z` - UTC datetime

## PMML ##

PMML defines four temporal data types for representing instants (points in time):

* `date` - Local date
* `datetime` - Local datetime
* `time` - Local time (24-hour clock)
* `timeSeconds` - Local time (unrestricted clock)

They are all "local" in a sense that they do not maintain explicit time zone offset (close analogy with `java.time.LocalDate`, `java.time.LocalDateTime` and `java.time.Time` Java classes).
If a predictive analytics application is dealing with temporal values associated with different time zones, then it should unify them to a common time zone (UTC or local time zone) before passing them on to the PMML engine.

Instants should be regarded as discrete.
The natural operational type is ordinal (ordered categorical), because it is possible to compare instants for equality plus determine the ordering between them (eg. answering a question "is instant A earlier/later than instant B?").

PMML further defines two sets of temporal data types for representing durations (distances between two points in time):

1. `dateDaysSince[<year>]` - Distance from the epoch in days
2. `dateTimeSecondsSince[<year>]` - Distance from the epoch in seconds

The epoch can take values `0`, `1960`, `1970` or `1980`.
The JPMML ecosystem extends this range with values `1990`, `2000`, `2010` and `2020` as proposed in [http://mantis.dmg.org/view.php?id=234](http://mantis.dmg.org/view.php?id=234).

Durations should be regarded as continuous integers.

PMML defines three built-in functions for converting instants to durations:

* `dateDaysSinceYear(x, <year>)`
* `dateSecondsSinceYear(x, <year>)`
* `dateSecondsSinceMidnight(x)`

The `dateDaysSinceYear` and `dateSecondsSinceYear` built-in functions support arbitrary epochs.
The suggestion is to use an epoch that would minimize the range of computed durations, and make them easier to analyze and explain for humans.
A good choice is the minimum year of the training dataset (restricts values to `[0, (maximum - minimum)]`).

In principle, subtracting one instant from another should yield a duration, and adding a duration to an instant should yield another instant.
The PMML specification does not clarify the behaviour of temporal values in the context of arithmetic operations so, while technically feasible, it should be avoided for the time being.

## Data pre-processing ##

The sample dataset is a list of crewed lunar missions under the [Apollo program](https://en.wikipedia.org/wiki/List_of_Apollo_missions).
In years 1968 thorugh 1972 there were nine flights.
The first two were lunar orbiting missions, and the remaining seven were lunar landing missions.

``` python
from pandas import DataFrame

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
```

In Python language speak, a datetime with time zone information is regarded as "(time zone-) offset-aware", whereas a datetime without time zone information is called a "(time zone-) offset-naive".
Python datetime functions typically raise an error when offset-aware and offset-naive datetimes are interacted:

``` python
from datetime import datetime, timezone

# Offset-aware begin date
begin = datetime(year = 2020, month = 1, day = 1, tzinfo = timezone.utc)

# Offset-naive end date
end = datetime(year = 2020, month = 3, day = 8, tzinfo = None)

# Raises a TypeError: "can't subtract offset-naive and offset-aware datetimes"
duration = end - begin
```

The aim of data pre-procesing is to convert offset-aware UTC datetimes to offset-naive PMML-compatible local datetimes:

``` python
import pandas

def awarestr_to_naivestr(x, tzinfo):
  # Parse string into offset-aware datetime
  x = pandas.to_datetime(x)
  # Unify time zones
  x = x.dt.tz_convert(tzinfo)
  # Convert from offset-aware to offset-naive
  x = x.dt.tz_localize(None)
  # Format offset-naive datetime into string
  x = x.dt.strftime("%Y-%m-%dT%H:%M:%S")

tzinfo = "Europe/Tallinn"

df["launch"] = awarestr_to_naivestr(df["launch"], tzinfo)
df["moon landing"] = awarestr_to_naivestr(df["moon landing"], tzinfo).replace({"NaT" : None})
df["return"] = awarestr_to_naivestr(df["return"], tzinfo)

print(df)
```

For example, the first cell of the dataset is converted from `1968-12-21T12:51:00Z` to `1968-12-21T15:51:00`.
The hour of day has been incremented by three hours (representing the time zone offset between UTC and Estonia/Tallinn time zone on 21st of December, 1968), and the `Z` suffix has been truncated.

## Feature specification and engineering ##

The `sklearn2pmml` package provides domain decorators and transformers for working with pre-processed temporal values.

Domain decorators are meant for declaring the type and behaviour of individual features.
They were discussed in detail in an earlier blog post about [extending Scikit-Learn with feature specifications]({% post_url 2020-02-23-sklearn_feature_specification_pmml %}).

The `sklearn2pmml.decoration` module provides three domain decorators:

* `TemporalDomain`
  * `DateDomain` - Default date
  * `DateTimeDomain` - Default datetime
* `OrdinalDomain` - Custom date or datetime

Domain decorators take care of parsing or casting input values to appropriate temporal values.

`TemporalDomain` decorators can be applied to multiple columns at once.
Their core configuration is hard-coded to prevent the collection and storage of valid value space information (ie. `Domain(with_data = False, with_statistics = False)`).
The main assumption is that the temporal feature(s) is likely to take previously unseen values.

In contrast, the `OrdinalDomain` decorator can only be applied to one column at once, but is fully configurable.
It may come in handy when a temporal feature must be restricted in a certain way.

A datetime is a complex data structure which needs to be "flattened" to a scalar before it can be fed to Scikit-Learn algorithms.
The `sklearn2pmml.preprocessing` module provides three transformers, which correspond to previously discussed PMML built-in functions:

* `DaysSinceYearTransformer`
* `SecondsSinceYearTransformer`
* `SecondsSinceMidnightTransformer`

Further transformations are possible using the good old `ExpressionTransformer` transformer.

For example, calculating the duration of a mission in seconds:

``` python
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import DateTimeDomain
from sklearn2pmml.preprocessing import ExpressionTransformer, SecondsSinceYearTransformer

mapper = DataFrameMapper([
  (["launch", "return"], [DateTimeDomain(), SecondsSinceYearTransformer(year = 1968), ExpressionTransformer("X[1] - X[0]")])
])
duration = mapper.fit_transform(df)

print(duration)
```

The lack of more fine-grained calendaring functions can be overcome by performing the required arithmetic operations manually.
If some functionality is needed often, then it should be extracted into a separate utility function.

For example, calculating the hour of day (24-hour clock) by dividing the number of seconds since midnight by 3600 seconds/hour:

``` python
from sklearn2pmml.decoration import Alias
from sklearn2pmml.preprocessing import SecondsSinceMidnightTransformer

import numpy

def make_hour_of_day_transformer():
  return ExpressionTransformer("numpy.floor(X[0] / 3600)")

mapper = DataFrameMapper([
  (["launch"], [DateTimeDomain(), SecondsSinceMidnightTransformer(), Alias(make_hour_of_day_transformer(), "HourOfLaunch", prefit = True)])
])
hour_of_day = mapper.fit_transform(df)

print(hour_of_day)
```

## Resources ##

* Python script: [`train.py`]({{ "/resources/2020-03-08/train.py" | absolute_url }})
