from xgboost import DMatrix

from util import *

import os
import pickle
import sys

cat_cols = ["Education", "Employment", "Gender", "Marital", "Occupation"]
cont_cols = ["Age", "Hours", "Income"]

cont_cols.remove("Income")

def measure_array(title, X):
	print("ndarray({})".format(title))

	print("\tX.__class__: {}".format(X.__class__))
	print("\tX.dtype: {}".format(X.dtype))
	# Dense array
	if hasattr(X, "nbytes"):
		print("\tX.nbytes: {}".format(X.nbytes))
	# Sparse array; assume CSR subtype
	else:
		print("\tX.csr_nbytes: {}".format(X.data.nbytes + X.indptr.nbytes + X.indices.nbytes))
	print("\tX.__sizeof__(): {}".format(X.__sizeof__()))
	print("\tlen(pickle.dumps(X)): {}".format(len(pickle.dumps(X))))

	print("")

def measure_dmatrix(title, dmat):
	print("DMatrix({})".format(title))

	print("\tdmat.__sizeof__(): {}".format(dmat.__sizeof__()))
	print("\tsys.getsizeof(dmat): {}".format(sys.getsizeof(dmat)))
	# ValueError: ctypes objects containing pointers cannot be pickled
	#print(len(pickle.dumps(dmat)))
	dmat.save_binary("dmatrix.bin", silent = False)
	print("\tlen(dmat.save_binary()): {}".format(os.path.getsize("dmatrix.bin")))

	print("")

#
# Dense dataset
#

df = load_audit(cat_cols, cont_cols)

# Cast to the minimum-size integer data type
cast_cols(df, cont_cols, numpy.uint8)

print("\n-- Dense: legacy --\n")

measure_array("df", df.values)

for dtype in [numpy.uint8, numpy.float32, numpy.float64]:
	transformer = make_dense_legacy_transformer(cat_cols, cont_cols, dtype = dtype)
	Xt = transformer.fit_transform(df)
	measure_array("sparse=False, dtype={}".format(dtype.__name__), Xt)

	dmat = DMatrix(data = Xt, label = df["Adjusted"])
	measure_dmatrix("dtype={}".format(dtype.__name__), dmat)

# DON'T DO THIS - sparse=True
for dtype in [numpy.uint8, numpy.float32, numpy.float64]:
	transformer = make_dense_legacy_transformer(cat_cols, cont_cols, sparse = True, dtype = numpy.uint8)
	Xt = transformer.fit_transform(df)
	measure_array("sparse=True, dtype={} # INVALID".format(numpy.uint8.__name__), Xt)

	dmat = DMatrix(data = Xt, label = df["Adjusted"])
	measure_dmatrix("dtype={} # INVALID".format(dtype.__name__), dmat)

# DON'T DO THIS - missing=0
if True:
	transformer = make_dense_legacy_transformer(cat_cols, cont_cols)
	Xt = transformer.fit_transform(df)

	dmat = DMatrix(data = Xt, label = df["Adjusted"], missing = 0)
	measure_dmatrix("missing=0, dtype={} # INVALID".format(numpy.uint8.__name__), dmat)

print("\n-- Dense: category --\n")

dmat = DMatrix(data = df[cat_cols + cont_cols], label = df["Adjusted"], enable_categorical = True)
measure_dmatrix("enable_categorical=True", dmat)

#
# Sparse dataset
#

df = load_audit_na(cat_cols, cont_cols)

# Cast to the minimum-size NA-enabled integer data type
cast_cols(df, cont_cols, pandas.UInt8Dtype())

print("\n-- Sparse: legacy --\n")

measure_array("df", df.values)

transformer = make_sparse_legacy_transformer(cat_cols, cont_cols)
Xt = transformer.fit_transform(df)
measure_array("category", Xt)

dmat = DMatrix(data = Xt, label = df["Adjusted"])
measure_dmatrix("category", dmat)

print("\n-- Sparse: category --\n")

# DMatrix does not support Pandas' extension types
cast_cols(df, cont_cols, numpy.float64)

dmat = DMatrix(data = df[cat_cols + cont_cols], label = df["Adjusted"], enable_categorical = True)
measure_dmatrix("enable_categorical=True", dmat)