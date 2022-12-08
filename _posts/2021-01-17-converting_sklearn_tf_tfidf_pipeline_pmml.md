---
layout: post
title: "Converting Scikit-Learn based TF(-IDF) pipelines to PMML documents"
author: vruusmann
keywords: scikit-learn sklearn2pmml tf-idf
---

The outline of a TF(-IDF) workflow:

1. Text tokenization.
2. Token normalization (case conversion, stemming, lemmatization).
3. Token filtering (removing stop words and low-importance words).
4. Token aggregation into terms (n-gram generation).
5. Term score estimation.

### Scikit-Learn

Typical implementation:

``` python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

import pandas

df = pandas.read_csv("sentiment.csv")

pipeline = Pipeline([
  ("tfidfvectorizer", TfidfVectorizer(stop_words = "english", max_features = 500, ngram_range = (1, 3), norm = None)),
  ("estimator", ...)
])
pipeline.fit(df["Sentence"], df["Score"])
```

Scikit-Learn packs TF(-IDF) workflow operations 1 through 4 into a single transformer - [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for TF, and [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) for TF-IDF:

1. Text tokenization is controlled using one of `tokenizer` or `token_pattern` attributes.
2. Token normalization is controlled using `lowercase` and `strip_accents` attributes.
3. Token filtering is controlled using `stop_words`, `min_df`, `max_df` and `max_features` attributes.
4. Token aggregation is controlled using the `ngram_range` attribute.

Term scores are estimated using the final estimator step.

Linear models estimate a score for each and every term.
The sentence score is the sum of its constituent term scores.
For better interpretability, it is advisable to keep sentences short and uniform (ie. sentences should parse into structurally similar token sets), and constrain the number of features.

Decision tree models estimate a score for combinations of terms.
The sentence score is the value associated with a decision path like "sentence contains term A, and does not contain terms B and C".
Decision trees can be ensembled either via bagging (random forest) or boosting (XGBoost, LightGBM), which gives them scoring properties that are more similar to linear models.

### PMML

The Predictive Model Markup Language (PMML) provides a [`TextIndex`](https://dmg.org/pmml/v4-4-1/Transformations.html#xsdElement_TextIndex) element for representing TF(-IDF) operations.
In brief, this tranformation takes a string input value, normalizes it, and then counts the occurrences of the specified term.
Term matching can be strict or fuzzy.

Text tokenization rules must be expressed in the form of regular expressions (REs).

The default behaviour for PMML (and Apache Spark ML) is *text splitting* using a *word separator RE*:

``` xml
<TextIndex textField="input" wordSeparatorCharacterRE="\s+">
  <Constant>term</Constant>
</TextIndex>
```

Splitting yields "dirty" tokens. They are automatically cleansed by trimming all the leading and trailing punctuation characters.

A splitting tokenizer is available as the `sklearn2pmml.feature_extraction.text.Splitter` callable type:

``` python
from sklearn2pmml.feature_extraction.text import Splitter

countvectorizer = CountVectorizer(tokenizer = Splitter(word_separator_re = "\s+"))
```

The default behaviour for Scikit-Learn is *token matching* (aka token extraction) using a *word RE*.
Unfortunately, this behaviour cannot be supported by the standard `wordSeparatorCharacterRE` attribute, because there is no straightforward way of translating between word and word separator REs.

The JPMML ecosystem extends the `TextIndex` element with the `x-wordRE` attribute as proposed in [http://mantis.dmg.org/view.php?id=271](http://mantis.dmg.org/view.php?id=271):

``` xml
<TextIndex textField="input" x-wordRE="\w+">
  <Constant>term</Constant>
</TextIndex>
```

Matching is assumed to yield "clean" tokens.
A data scientist shall be free to craft a word RE that extracts and retains significant punctuation or whitespace characters.

A matching tokenizer is available as the `sklearn2pmml.feature_extraction.text.Matcher` callable type:

``` python
from sklearn2pmml.feature_extraction.text import Matcher

countvectorizer = CountVectorizer(tokenizer = Matcher(word_re = "\w+"))
```

Another difference between TF(-IDF) workflows is that PMML performs text normalization (precedes tokenization) whereas Scikit-Learn performs token normalization (follows tokenization).

Text normalization is activated by adding one or more [`TextIndexNormalization`](https://dmg.org/pmml/v4-4-1/Transformations.html#xsdElement_TextIndexNormalization) child elements to the `TextIndex` element.
Again, the rules must be expressed in the form of regular expressions.

The JPMML-SkLearn library currently uses a single `TextIndexNormalization` element for encoding the removal of stop words.
Future versions may use more to encode stemming, lemmatization and other string manipulations.

Alternatively, stemming and lemmatization can be emulated by manually specifying the `maxLevenshteinDistance` attribute.
[Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) is metric that reflects the distance between two character sequences in terms of the minimum number of one-character edits (additions, replacements or removals).

For example, in the English language, the Levenshtein distance between the singular and plural forms of a regular noun is 1 (ie. the "s" suffix). Knowing this, it is trivial to make one `TextIndex` element match both forms:

``` xml
<TextIndex textField="input" maxLevenshteinDistance="1" x-wordRE="\w+">
  <Constant>term</Constant>
</TextIndex>
```

Token filtering by importance and token aggregation do not require any PMML integration, because they are solely training-time phenomena.