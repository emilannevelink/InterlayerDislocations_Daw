from distutils.core import setup, Extension
import numpy.distutils.misc_util

#21 bring displacement back to first quadrant

setup(
    ext_modules=[Extension("_IDDAWt0", ["_cIDDAWt0.cpp", "IDDAWt0.cpp"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)


# Build using: CC=g++ python setup_ID_DAW.py build_ext --inplace