---
layout: post
title: "Applying ML models to database data: the REST web service approach"
author: vruusmann
keywords: openscoring
---

There are three major approaches to applying ML models to database data:

1. Direct SQL execution. This is attained by converting the model from its native representation to the SQL representation. For example, there are tools like [pmml2sql](http://www.pmml2sql.com/) and KNIME (version 2.11.1 and newer) that claim to have the ability to convert most common model types from PMML to SQL. Naturally, the quality of conversion work varies considerably between different SQL dialects.
2. Intermediated SQL execution. The model stays in its native representation. The evaluation is handled by a dedicated model evaluation engine that is tightly integrated into the database backend. For example, PostgreSQL database supports the execution of arbitrary R and Python application code via [PL/R](https://www.joeconway.com/plr/) and [PL/Py](http://python.projects.postgresql.org/) procedural languages, respectively. This approach is technically quite demanding, because it crosses SQL and application programming domains. The life of SQL end users can be made somewhat easier by (automatically-) generating an appropriate SQL wrapper function for every model.
3. External execution. The model is deployed to a dedicated model evaluation engine that is separate from the database backend. Such model evaluation engine could be shared between several applications and services, which leads to the concept of "organization's predictive analytics hub".

The choice between these three approaches depends on various technical and organizational considerations. Fundamentally, direct and intermediated SQL execution approaches are about **moving the model to where data are located**, whereas the external execution approach is about **moving data to where the model is located**. SQL execution approaches can operate in real-time on any amount of data. The external execution approach is penalized by the REST web service overhead and network round-trip times. This penalty scales sublinearly. Therefore, it becomes less of an issue if the evaluation operations are less frequent (real-time vs. batch queries) and deal with larger amounts of data (single data record vs. collection of data records). The main advantage of the external execution approach is that is easily applicable to any database backend.

### Overview

This blog post demonstrates how to use the [Openscoring REST web service](https://github.com/openscoring/openscoring) from within PostgreSQL database.

The exercise starts with installing and running the Openscoring web service at localhost:8080 and deploying the example model `DecisionTreeIris.pmml` as described in the README file of the project.

Every deployed model should be verified by accessing its summary REST API endpoint. For example, the summary of the newly deployed `DecisionTreeIris` model can be downloaded as a JSON object by performing an HTTP GET request on [http://localhost:8080/openscoring/model/DecisionTreeIris](http://localhost:8080/openscoring/model/DecisionTreeIris).

The `schema` attribute of this summary object contains the description of the model schema. The `activeFields` and `groupFields` attributes of the schema object represent model arguments, whereas the `targetFields` and `outputFields` attributes represent model results. It is easy to see that the `DecisionTreeIris` model expects four arguments "Sepal\_Length", "Sepal\_Width", "Petal\_Length" and "Petal\_Width", and produces six results "Species", "Predicted\_Species", "Probability\_setosa", "Probability\_versicolor", "Probability\_virginica" and "Node\_Id".

The external execution workflow contains three steps:

1. Exporting data from database to a CSV document `/tmp/iris_request.csv`.
2. Performing the model evaluation by calling the CSV prediction REST API endpoint.
3. Importing data from a CSV document `/tmp/iris_response.csv` back to database.

### Data schema

The data about iris flowers is separated into two tables based on their origin. First, the table `iris` contains experimentally determined data. It is populated with 150 data records from the example file `/tmp/input.csv`. Second, the table `iris_decisiontree` contains predicted data:

``` sql
CREATE TABLE iris (
  Id serial PRIMARY KEY,
  Sepal_Length double precision,
  Sepal_Width double precision,
  Petal_Length double precision,
  Petal_Width double precision,
  Species varchar
);

COPY iris(Sepal_Length, Sepal_Width, Petal_Length, Petal_Width, Species) FROM '/tmp/input.csv' WITH CSV DELIMITER ',' HEADER;

CREATE TABLE iris_decisiontree (
  Id integer REFERENCES iris(Id),
  Predicted_Species varchar,
  Probability_setosa double precision,
  Probability_versicolor double precision,
  Probability_virginica double precision
);
```

This separation makes it straightforward to scale the application from one model to many models. For example, if it becomes necessary to deploy an alternative model `RandomForestIris`, then these predicted data will be stored in another table `iris_randomforest`. A random forest model is a collection of decision tree models. The results take longer to compute, but should be more accurate.

### Export of model argument data

The table `iris` is (periodically-) monitored for data records that do not have a counterpart in the table `iris_decisiontree`. All such unclassified data records are exported to a CSV document `/tmp/iris_request.csv` using the `COPY .. TO` command:

``` sql
COPY (SELECT iris.Id AS "Id", iris.Sepal_Length AS "Sepal_Length", iris.Sepal_Width AS "Sepal_Width", iris.Petal_Length AS "Petal_Length", iris.Petal_Width AS "Petal_Width" FROM iris LEFT JOIN iris_decisiontree ON iris.Id = iris_decisiontree.Id WHERE iris_decisiontree.Predicted_Species IS NULL) TO '/tmp/iris_request.csv' WITH CSV DELIMITER ',' HEADER;
```

The CSV document must conform to the following rules:

* The first row ("header") represents column identifiers.
* The first column represents row identifiers. The name of this column must be "Id" (case-insensitive). The Openscoring web service copies the row identifier column from the request CSV document to the response CSV document without changes.
* All other columns represent field data. The ordering of field columns is not significant. The names of field columns are case sensitive. For example, the Openscoring web service throws an exception if the model expects a "Sepal\_Length" column as an argument, but the CSV document provides a "sepal\_length" column instead. Most database backends treat column names in a case-insensitive manner and automatically standardize them to the lowercase form. The above `COPY .. TO` command fights this behaviour by specifying the correct case using the `AS` alias.
* The separator character can be the comma (`,`), the semicolon (`;`) or the horizontal tab (`\t`) character. String values may be quoted using the double quote (`"`) character.
* Two consecutive separator characters (`,,`) indicate a missing field value. An empty string can be represented using two consecutive double quote characters (`,"",`).

### Model evaluation

The evaluation is handled by the Openscoring web service over the CSV evaluation REST API endpoint. In brief, this REST API endpoint is bound to the HTTP POST method. The request body is a CSV document with model arguments. The request is processed synchronously. For better responsiveness, application clients can perform the evaluation in parallel, where one big request is split into several smaller requests. The response body is another CSV document with model results.

Database engines typically do not advertise HTTP client functionality as their core competency. It becomes a one-time responsibility for SQL developers to find and install a suitable database extension for this purpose. In this exercise, the HTTP client functionality is provided by the [cURL](https://curl.se/) command-line application, which is executed from within PostgreSQL database using the [PL/sh](https://github.com/petere/plsh) procedural language.

Activating the PL/sh extension and creating two shell-backed SQL functions `evaluate_iris()` and `clean_iris()`:

``` sql
CREATE EXTENSION plsh;

-- uploads file `/tmp/iris_request.csv` to the Openscoring web service and downloads the result into file `/tmp/iris_response.csv`.
CREATE FUNCTION evaluate_iris() RETURNS void AS '
#!/bin/sh
curl -X POST --data-binary @/tmp/iris_request.csv -H "Content-type: text/plain" http://localhost:8080/openscoring/model/DecisionTreeIris/csv > /tmp/iris_response.csv
' LANGUAGE plsh;

-- deletes data exchange files `/tmp/iris_request.csv` and `/tmp/iris_response.csv`
CREATE FUNCTION clean_iris() RETURNS void AS '
#!/bin/sh
rm /tmp/iris_request.csv
rm /tmp/iris_response.csv
' LANGUAGE plsh;
```

SQL end users can then perform model evaluation by calling the `evaluate_iris()` function:

``` sql
SELECT evaluate_iris();
```

### Import of model result data

Class labels and associated probabilities are imported from the file `/tmp/iris_response.csv` to the table `iris_decisiontree` using the already familiar `COPY .. FROM` command.

The CSV document contains the row identifier column ("Id") and six field columns. However, the target table `iris_decisiontree` contains only four field columns. PostgreSQL database does not support copying only a subset of columns. Therefore, it becomes necessary to implement the import of classification results as a two step process. First, all field columns are copied to a temporary table `iris_decisiontree_temp`. Later, four field columns ("Predicted\_Species", "Probability\_setosa", "Probability\_versicolor" and "Probability\_virginica") are copied over to the target table `iris_decisiontree`, whereas two field columns ("Species" and "Node\_Id") are left behind.

``` sql
CREATE TEMPORARY TABLE iris_decisiontree_temp (
  Id integer,
  Species varchar,
  Predicted_Species varchar,
  Probability_setosa double precision,
  Probability_versicolor double precision,
  Probability_virginica double precision,
  Node_Id varchar
);

COPY iris_decisiontree_temp FROM '/tmp/iris_response.csv' WITH CSV DELIMITER ',' HEADER;

INSERT INTO iris_decisiontree(Id, Predicted_Species, Probability_setosa, Probability_versicolor, Probability_virginica) SELECT Id, Predicted_Species, Probability_setosa, Probability_versicolor, Probability_virginica FROM iris_decisiontree_temp;

DROP TABLE iris_decisiontree_temp;

SELECT clean_iris();
```

### Making use of classification results

Displaying all iris flowers where the experimentally determined species (column `iris.Species`) and the predicted species (column `iris_decisiontree.Predicted_Species`) are not equal:

``` sql
SELECT iris.Id, iris.Species, iris_decisiontree.Predicted_Species, iris_decisiontree.Probability_setosa, iris_decisiontree.Probability_versicolor, iris_decisiontree.Probability_virginica FROM iris LEFT JOIN iris_decisiontree ON iris.Id = iris_decisiontree.Id WHERE iris.Species != iris_decisiontree.Predicted_Species;
```
