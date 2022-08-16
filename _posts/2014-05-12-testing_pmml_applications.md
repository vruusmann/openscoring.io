---
layout: post
title: "Testing PMML applications"
author: vruusmann
keywords: jpmml-evaluator testing
---

The [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library aims to provide high quality service to its users. The main module contains unit tests that ensure compliance with the PMML specification. Additionally, there are several support modules that contain integration tests that ensure interoperability with popular third-party PMML converters such as [R/Rattle](https://rattle.togaware.com/), [KNIME](https://knime.com/) and [RapidMiner](https://rapidminer.com/).

However, there can never be too much testing. Application developers are encouraged to create and maintain custom integration test modules that replicate models and datasets from their production environments. Such integration tests lower the risk of change. They make it more secure to update and upgrade the ML framework and the PMML layer on top of it.

This blog post details a batch evaluation method for integration testing purposes. The heavy-lifting is handled by the `org.jpmml.evaluator.BatchUtil#difference(Batch, double, double)` method. A test case, which is represented by the `org.jpmml.evaluator.Batch` interface, is a triplet of streaming resources:

* PMML.
* Input CSV. Contains active and group field(s) as specified by the `MiningSchema` element.
* Output CSV. Contains target and output field(s) as specified by the `MiningSchema` and `Output` elements.

The JPMML-Evaluator library packages and distributes test classes as a separate JAR file. It can be included into Apache Maven builds using the following dependency declaration:

``` xml
<dependency>
  <groupId>org.jpmml</groupId>
  <artifactId>pmml-evaluator</artifactId>
  <version>${jpmml.version}</version>
  <type>test-jar</type>
  <scope>test</scope>
</dependency>
```

Under the hood, the batch utility class loads data records from the input CSV resource, evaluates them using the PMML resource and compares the resulting "actual output" data records against the "expected output" data records that were loaded from the output CSV resource. The method returns a list of differences between actual and expected output data records. A test is considered to be successful when the list of differences is empty.

CSV resources must have column identifiers (i.e. the header row), because the mapping between PMML fields and CSV data table columns is by name. In contrast, CSV resources must not have row identifiers, because the mapping between input and output CSV data table rows is by position. The field delimiter character is the comma. Missing field values are indicated by string constants "NA" or "N/A" (without quotes).

The `org.jpmml.evaluator.ArchiveBatch` class loads resources from the current Java Archive (JAR) file. A batch job is defined by a model identifier and a dataset identifier. These identifiers determine the locations of associated resources ("conventions over configuration"):

* PMML. `/pmml/<model identifier><dataset identifier>.pmml`
* Input CSV. `/csv/<dataset identifier>.csv`
* Output CSV. `/csv/<model identifier><dataset identifier>.csv`

The following R script creates a decision tree model for the "iris" dataset. The model identifier is "DecisionTree" and the dataset identifier is "Iris". All file paths are prefixed with `src/test/resources`, which is the root directory for test resources in Apache Maven builds.

``` r
library("pmml")
library("rpart")

data("iris")

# Generate input CSV
# The data table must contain four columns for active fields "Sepal.Length", "Sepal.Width", "Petal.Length" and "Petal.Width"
irisInput = iris[, 1:4]
write.table(irisInput, file = "src/test/resources/csv/Iris.csv", col.names = TRUE, row.names = FALSE, sep = ",", quote = FALSE)

iris.rpart = rpart(Species ~ ., iris)

# Generate PMML
saveXML(pmml(iris.rpart), file = "src/test/resources/pmml/DecisionTreeIris.pmml")

iris.class = predict(iris.rpart, newdata = iris, type = "class")
iris.prob = predict(iris.rpart, newdata = iris, type = "prob")

# Generate output CSV
# The data table must contain one column for the target field "Species" and four columns for output fields "Predicted_Species", "Probability_setosa", "Probability_versicolor" and "Probability_virginica"
irisOutput = data.frame(iris.class, iris.class, iris.prob)
names(irisOutput) = c("Species", "Predicted_Species", "Probability_setosa", "Probability_versicolor", "Probability_virginica")
write.table(irisOutput, file = "src/test/resources/csv/DecisionTreeIris.csv", col.names = TRUE, row.names = FALSE, sep = ",", quote = FALSE)
```

The generated decision tree model `DecisionTreeIris.pmml` contains a single target field and four output fields. The first output field "Predicted\_Species" is simply a copy of the target field, whereas the remaining three output fields "Probability\_setosa", "Probability\_versicolor" and "Probability\_virginica" give the probabilities for each target category. A thorough test handler will want to check all five fields (a not so thorough test handler could comment out or remove the `Output` element and check only the target field). The `predict.rpart` function of the [`rpart`](https://cran.r-project.org/package=rpart) package is executed twice in order to compile the necessary data table. The first execution (`type = "class"`) predicts class labels. The second execution (`type = "prob"`) computes the associated probabilities.

The following Java source code runs this batch job using the [JUnit framework](https://junit.org/):

``` java
package org.jpmml.example;

import java.util.List;

import com.google.common.collect.MapDifference;

import org.dmg.pmml.FieldName;

import org.jpmml.evaluator.ArchiveBatch;
import org.jpmml.evaluator.Batch;
import org.jpmml.evaluator.BatchUtil;

import org.junit.Assert;
import org.junit.Test;

public class ClassificationTest {

  @Test
  public void evaluateDecisionTreeIris() throws Exception {
    // The ArchiveBatch class is abstract, and needs to be instantiated as an anonymous inner class
    Batch batch = new ArchiveBatch("DecisionTree", "Iris"){};

    List<MapDifference<FieldName, ?>> differences = BatchUtil.difference(batch, 1.e-6, 1.e-6);
    if(!differences.isEmpty()){
      System.err.println(differences);

      Assert.fail();
    }
  }
}
```

The batch utility class performs value comparisons according to the [model verification](http://www.dmg.org/v4-3/ModelVerification.html) principles. In brief, categorical and ordinal field values must match exactly, whereas continuous field values must fall within the acceptable range. The range checking algorithm is implemented in the `org.jpmml.evaluator.VerificationUtil` utility class. It is controlled by two parameters `precision` and `zeroThreshold`. The acceptable range is defined relative to the expected value. The actual value is acceptable if it satisfies the condition: `(expected value * (1 - precision)) <= actual value <= (expected value * (1 + precision))`. This approach becomes numerically unstable when the expected value is zero or very close to it. In such case the acceptable range is defined in absolute terms. The condition becomes: `-1 * zeroThreshold <= actual value <= zeroThreshold`.

The above Java source code specifies both parameters as 1.e-6 (that is, one part per million or 0.000001, respectively). This batch job can be broken for demonstration purposes by changing the value of the "Probability\_setosa" field of the first expected output record from "1" to "0.9999989". The failure message is the following:

```
[not equal: value differences={Probability_setosa=(0.9999989, 1.0)}]
```

### Resources

* Demo Apache Maven project: [`DecisionTreeIris.zip`]({{ site.baseurl }}/assets/2014-05-12/DecisionTreeIris.zip)
