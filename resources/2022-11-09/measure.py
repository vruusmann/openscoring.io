from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import numpy
import types

X, y = make_regression(n_samples = 10000, n_features = 10)

#estimator = DecisionTreeRegressor(random_state = 13)
estimator = RandomForestRegressor(n_estimators = 31, random_state = 13)

def is_instance_attr(obj, name):
	if not hasattr(obj, name):
		return False
	if name.startswith("__") and name.endswith("__"):
		return False
	v = getattr(obj, name)
	if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType, types.MethodType)):
		return False
	# See https://stackoverflow.com/a/17735709/
	attr_type = getattr(type(obj), name, None)
	if isinstance(attr_type, property):
		return False
	return True

def get_instance_attrs(obj):
	names = dir(obj)
	names = [name for name in names if is_instance_attr(obj, name)]
	return names

def _qualname(clazz):
	return ".".join([clazz.__module__, clazz.__name__])

def deep_sklearn_sizeof(obj, verbose = True):
	# Primitive-valued attributes
	if obj is None:
		return obj.__sizeof__()
	elif isinstance(obj, (int, float, str, bool, numpy.int64, numpy.float32, numpy.float64)):
		return obj.__sizeof__()
	# Iterable attributes
	elif isinstance(obj, list):
		sum = [].__sizeof__() # Empty list
		for v in obj:
			v_sizeof = deep_sklearn_sizeof(v, verbose = False)
			sum += v_sizeof
		return sum
	elif isinstance(obj, tuple):
		sum = ().__sizeof__() # Empty tuple
		for i, v in enumerate(obj):
			v_sizeof = deep_sklearn_sizeof(v, verbose = False)
			sum += v_sizeof
		return sum
	# Numpy ndarray-valued attributes
	elif isinstance(obj, numpy.ndarray):
		sum = obj.__sizeof__() # Array container
		sum += (obj.size * obj.itemsize) # Array content
		return sum
	else:
		qualname = _qualname(obj.__class__)
		# Restrict the circle of competence to Scikit-Learn classes
		if not (qualname.startswith("_abc.") or qualname.startswith("sklearn.")):
			raise ValueError(qualname)
		sum = object().__sizeof__() # Empty object
		names = get_instance_attrs(obj)
		if names:
			if verbose:
				print("| Attribute | `type(v)` | `deep_sklearn_sizeof(v)` |")
				print("|---|---|---|")
			for name in names:
				v = getattr(obj, name)
				v_type = type(v)
				v_sizeof = deep_sklearn_sizeof(v, verbose = False)
				sum += v_sizeof
				if verbose:
					print("| {} | {} | {} |".format(name, v_type, v_sizeof))
		return sum

print("Initial state:")
print("{} B\n".format(deep_sklearn_sizeof(estimator)))

estimator.fit(X, y)

print("Final fitted state:")
print("{} B\n".format(deep_sklearn_sizeof(estimator)))
	
if isinstance(estimator, DecisionTreeRegressor):
	tree = regressor.tree_
elif isinstance(estimator, RandomForestRegressor):
	tree = estimator.estimators_[0].tree_
else:
	raise ValueError()

print("Tree state:")
print("{} B\n".format(deep_sklearn_sizeof(tree)))