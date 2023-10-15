from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn2pmml import load_class_mapping, make_class_mapping_jar, sklearn2pmml
from sklearn2pmml.statsmodels import StatsModelsRegressor
from sklearn2pmml.util import fqn
from statsmodels.api import GLM, OLS, WLS
from statsmodels.genmod import families

import joblib
import pandas

df = pandas.read_csv("auto.csv")

# Shuffle rows to make the CV part of GridSearchCV perform better
df = df.sample(frac = 1)

cat_cols = ["cylinders", "model_year", "origin"]
cont_cols = ["acceleration", "displacement", "horsepower", "weight"]

X = df[cat_cols + cont_cols]
y = df["mpg"]

def make_statsmodels_pipeline(regressor):
	transformer = ColumnTransformer([
		("cat", OneHotEncoder(drop = "first", handle_unknown = "infrequent_if_exist", sparse_output = False), cat_cols),
		("cont", StandardScaler(), cont_cols)
	])

	return Pipeline([
		("transformer", transformer),
		("regressor", regressor)
	])

#
# Model selection
#

pipeline = make_statsmodels_pipeline(StatsModelsRegressor(OLS))

ctor_params_grid = {
	"regressor__model_class" : [GLM, OLS, WLS],
	"regressor__fit_intercept" : [True, False]
}

tuner = GridSearchCV(pipeline, param_grid = ctor_params_grid, verbose = 3)
tuner.fit(X, y, regressor__fit_method = "fit")

print(tuner.best_estimator_)
print(tuner.best_score_)

#
# Model selection with model hyperparameter tuning
#

"""
class TunableStatsModelsRegressor(StatsModelsRegressor):

	def __init__(self, model_class, fit_intercept = True, alpha = 0.01, L1_wt = 1, **init_params):
		super(TunableStatsModelsRegressor, self).__init__(model_class = model_class, fit_intercept = fit_intercept, **init_params)
		self.alpha = alpha
		self.L1_wt = L1_wt

	def fit(self, X, y, **fit_params):
		super(TunableStatsModelsRegressor, self).fit(X, y, alpha = self.alpha, L1_wt = self.L1_wt, **fit_params)
		return self
"""

class TunableStatsModelsRegressor(StatsModelsRegressor):

	def __init__(self, model_class, fit_intercept = True, tune_params = {}, **init_params):
		super(TunableStatsModelsRegressor, self).__init__(model_class = model_class, fit_intercept = fit_intercept, **init_params)
		self.tune_params = tune_params

	def set_params(self, **params):
		super_params = dict([(k, params.pop(k)) for k, v in dict(**params).items() if k in ["model_class", "fit_intercept", "tune_params"]])
		super(TunableStatsModelsRegressor, self).set_params(**super_params)
		setattr(self, "tune_params", dict(**params))

	def fit(self, X, y, **fit_params):
		super(TunableStatsModelsRegressor, self).fit(X, y, **self.tune_params, **fit_params)
		return self

#pipeline = make_statsmodels_pipeline(StatsModelsRegressor(OLS))
pipeline = make_statsmodels_pipeline(TunableStatsModelsRegressor(OLS))

ctor_params_grid = {
	"regressor__fit_intercept" : [True, False]
}

regfit_params_grid = {
	"regressor__alpha" : loguniform(1e-2, 1).rvs(5),
	"regressor__L1_wt" : uniform(0, 1).rvs(5)
}

tuner = GridSearchCV(pipeline, param_grid = {**ctor_params_grid, **regfit_params_grid}, verbose = 3)
tuner.fit(X, y, regressor__fit_method = "fit_regularized")

print(tuner.best_estimator_)
print(tuner.best_score_)

#
# Model class conversion and persistence in pickle data format
#

"""
best_pipeline = tuner.best_estimator_

best_regressor = best_pipeline._final_estimator
best_regressor.__class__ = StatsModelsRegressor

joblib.dump(best_pipeline, "GridSearchAuto.pkl")
"""

best_params = dict(tuner.best_params_)

best_pipeline = make_statsmodels_pipeline(StatsModelsRegressor(OLS, fit_intercept = best_params.pop("regressor__fit_intercept")))
best_pipeline.fit(X, y, **best_params, regressor__fit_method = "fit_regularized")

joblib.dump(best_pipeline, "GridSearchAuto.pkl")

#
# Model persistence in PMML data format
#

default_mapping = load_class_mapping()

statsmodels_mapping = {
	fqn(TunableStatsModelsRegressor) : default_mapping[fqn(StatsModelsRegressor)]
}

extension_jar = "TunableStatsModelsRegressor.jar"

make_class_mapping_jar(statsmodels_mapping, extension_jar)

sklearn2pmml(tuner.best_estimator_, "GridSearchAuto.pmml", user_classpath = [extension_jar])