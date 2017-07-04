from binomial cimport binomial_t
cimport numpy as np
cimport cython
from libc cimport string
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t, intptr_t)
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cpython cimport Py_INCREF, PyComplex_FromDoubles
from cython_overrides cimport PyFloat_AsDouble, PyInt_AsLong, PyComplex_RealAsDouble, PyComplex_ImagAsDouble
from distributions cimport aug_state

cdef class RandomState:

    cdef void *rng_loc
    cdef binomial_t binomial_info
    cdef aug_state rng_state
    cdef object lock
    cdef object __seed
    cdef object __stream
    cdef object __version

    cdef double c_standard_normal(self)
    cdef double c_random_sample(self)
    cdef inline _shuffle_raw(self, np.npy_intp n, np.npy_intp itemsize, np.npy_intp stride, char* data, char* buf)
