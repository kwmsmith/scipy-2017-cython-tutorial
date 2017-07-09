#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
cimport cython
import sys
import operator
import warnings
try:
    from threading import Lock
except:
    from dummy_threading import Lock
import numpy as np
import rng

from distributions cimport *
include "rk_state_len.pxi"

np.import_array()

cdef object _get_state(aug_state state):
    cdef uint32_t [:] key = np.zeros(RK_STATE_LEN, dtype=np.uint32)
    cdef Py_ssize_t i
    for i in range(RK_STATE_LEN):
        key[i] = state.rng.key[i]
    return (np.asanyarray(key), state.rng.pos)

cdef object _set_state(aug_state *state, object state_info):
    cdef uint32_t [:] key = state_info[0]
    cdef Py_ssize_t i
    for i in range(RK_STATE_LEN):
        state.rng.key[i] = key[i]
    state.rng.pos = state_info[1]

# __normal_method = u'zig'
__normal_method = u'bm'


include "array_utilities.pxi"
include "array_fillers.pxi"
include "bounded_integers.pxi"
include "aligned_malloc.pxi"

cdef extern from "distributions.h":

    cdef uint64_t random_uint64(aug_state* state) nogil
    cdef uint32_t random_uint32(aug_state* state) nogil
    cdef uint64_t random_raw_values(aug_state* state) nogil

    cdef long random_positive_int(aug_state* state) nogil
    cdef unsigned long random_uint(aug_state* state) nogil
    cdef unsigned long random_interval(aug_state* state, unsigned long max) nogil

    cdef float random_standard_uniform_float(aug_state* state) nogil
    cdef float random_gamma_float(aug_state *state, double shape, float scale) nogil
    cdef float random_standard_gamma_float(aug_state* state, float shape) nogil

    cdef double random_standard_uniform_double(aug_state* state) nogil
    cdef double random_gauss(aug_state* state) nogil
    cdef double random_gauss_zig(aug_state* state) nogil
    cdef double random_gauss_zig_julia(aug_state* state) nogil
    cdef double random_standard_exponential(aug_state* state) nogil
    cdef double random_standard_cauchy(aug_state* state) nogil

    cdef double random_exponential(aug_state *state, double scale) nogil
    cdef double random_standard_gamma(aug_state* state, double shape) nogil
    cdef double random_pareto(aug_state *state, double a) nogil
    cdef double random_weibull(aug_state *state, double a) nogil
    cdef double random_power(aug_state *state, double a) nogil
    cdef double random_rayleigh(aug_state *state, double mode) nogil
    cdef double random_standard_t(aug_state *state, double df) nogil
    cdef double random_chisquare(aug_state *state, double df) nogil

    cdef double random_normal(aug_state *state, double loc, double scale) nogil
    cdef double random_normal_zig(aug_state *state, double loc, double scale) nogil
    cdef double random_uniform(aug_state *state, double loc, double scale) nogil
    cdef double random_gamma(aug_state *state, double shape, double scale) nogil
    cdef double random_beta(aug_state *state, double a, double b) nogil
    cdef double random_f(aug_state *state, double dfnum, double dfden) nogil
    cdef double random_laplace(aug_state *state, double loc, double scale) nogil
    cdef double random_gumbel(aug_state *state, double loc, double scale) nogil
    cdef double random_logistic(aug_state *state, double loc, double scale) nogil
    cdef double random_lognormal(aug_state *state, double mean, double sigma) nogil
    cdef double random_noncentral_chisquare(aug_state *state, double df, double nonc) nogil
    cdef double random_wald(aug_state *state, double mean, double scale) nogil
    cdef double random_vonmises(aug_state *state, double mu, double kappa) nogil

    cdef double random_noncentral_f(aug_state *state, double dfnum, double dfden, double nonc) nogil
    cdef double random_triangular(aug_state *state, double left, double mode, double right) nogil

    cdef long random_poisson(aug_state *state, double lam) nogil
    cdef long random_negative_binomial(aug_state *state, double n, double p) nogil
    cdef long random_binomial(aug_state *state, double p, long n) nogil
    cdef long random_logseries(aug_state *state, double p) nogil
    cdef long random_geometric(aug_state *state, double p) nogil
    cdef long random_zipf(aug_state *state, double a) nogil
    cdef long random_hypergeometric(aug_state *state, long good, long bad, long sample) nogil

    cdef void random_bounded_uint64_fill(aug_state *state, uint64_t off, uint64_t rng, intptr_t cnt, uint64_t *out) nogil
    cdef void random_bounded_uint32_fill(aug_state *state, uint32_t off, uint32_t rng, intptr_t cnt,uint32_t *out) nogil
    cdef void random_bounded_uint16_fill(aug_state *state, uint16_t off, uint16_t rng, intptr_t cnt, uint16_t *out) nogil
    cdef void random_bounded_uint8_fill(aug_state *state, uint8_t off, uint8_t rng, intptr_t cnt, uint8_t *out) nogil
    cdef void random_bounded_bool_fill(aug_state *state, np.npy_bool off, np.npy_bool rng, intptr_t cnt, np.npy_bool *out) nogil

    cdef void random_gauss_zig_float_fill(aug_state *state, intptr_t count, float *out) nogil
    cdef void random_uniform_fill_float(aug_state *state, intptr_t cnt, double *out) nogil
    cdef void random_standard_exponential_fill_float(aug_state* state, intptr_t count, float *out) nogil
    cdef void random_gauss_fill_float(aug_state* state, intptr_t count, float *out) nogil

    cdef void random_gauss_zig_double_fill(aug_state* state, intptr_t count, double *out) nogil
    cdef void random_uniform_fill_double(aug_state *state, intptr_t cnt, double *out) nogil
    cdef void random_standard_exponential_fill_double(aug_state* state, intptr_t count, double *out) nogil
    cdef void random_gauss_fill(aug_state* state, intptr_t count, double *out) nogil
    cdef void random_gauss_zig_julia_fill(aug_state* state, intptr_t count, double *out) nogil


cdef class RandomState:
    """
    RandomState(seed=None)

    Container for the Mersenne Twister pseudo-random number generator.
    """

    poisson_lam_max = POISSON_LAM_MAX
    __MAXSIZE = <uint64_t>sys.maxsize

    def __init__(self, seed=None):
        self.rng_state.rng = <rng_t *>PyArray_malloc_aligned(sizeof(rng_t))
        self.rng_state.binomial = &self.binomial_info
        self.lock = Lock()
        self.__version = 0

        self.__seed = seed
        self.__stream = None

        self._reset_state_variables()
        self.seed(seed)

    def __dealloc__(self):
        PyArray_free_aligned(self.rng_state.rng)

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        This method is called when ``RandomState`` is initialized. It can be
        called again to re-seed the generator. For details, see ``RandomState``.

        Parameters
        ----------
        seed : int or array_like, optional
            Seed for ``RandomState``.
            Must be convertible to 32 bit unsigned integers.

        See Also
        --------
        RandomState

        """
        cdef np.ndarray obj
        if seed is None:
            seed = 1
        try:
            if hasattr(seed, 'squeeze'):
                seed = seed.squeeze()
            idx = operator.index(seed)
            if idx > int(2**32 - 1) or idx < 0:
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            with self.lock:
                set_seed(&self.rng_state, seed)
        except TypeError:
            obj = np.asarray(seed).astype(np.int64, casting='safe')
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            obj = obj.astype(np.uint32, casting='unsafe', order='C')
            with self.lock:
                set_seed_by_array(&self.rng_state,
                                  <uint32_t*> obj.data,
                                  np.PyArray_DIM(obj, 0))
        self._reset_state_variables()

    def _reset_state_variables(self):
        self.rng_state.gauss = 0.0
        self.rng_state.gauss_float = 0.0
        self.rng_state.has_gauss = 0
        self.rng_state.has_gauss_float = 0
        self.rng_state.has_uint32= 0
        self.rng_state.uinteger = 0
        self.rng_state.binomial.has_binomial = 0

    # Basic distributions:

    cdef double c_random_sample(self):
        return double_fill(&self.rng_state,
                           &random_uniform_fill_double,
                           None,
                           self.lock,
                           None)


    def random_sample(self, size=None, dtype=np.float64, out=None):
        """
        random_sample(size=None, dtype='d', out=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random_sample() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result. All dtypes are determined by their
            name, either 'float64' or 'float32'. The default value is
            'float64'.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        Examples
        --------
        >>> np.random.random_sample()
        0.47108547995356098
        >>> type(np.random.random_sample())
        <type 'float'>
        >>> np.random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        key = np.dtype(dtype).name
        if key == 'float64':
            return double_fill(&self.rng_state, &random_uniform_fill_double, size, self.lock, out)
        elif key == 'float32':
            return float_fill(&self.rng_state, &random_uniform_fill_float, size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype "%s" for random_sample' % key)


    # Complicated, continuous distributions:
    cdef double c_standard_normal(self):
        return double_fill(&self.rng_state,
                           &random_gauss_zig_double_fill,
                           None,
                           self.lock,
                           None)

    def standard_normal(self, size=None, dtype=np.float64, method=__normal_method,
                        out=None):
        """
        standard_normal(size=None, dtype='d', method='bm', out=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result. All dtypes are determined by their
            name, either 'float64' or 'float32'. The default value is
            'float64'.
        method : str, optional
            Either 'bm' or 'zig'. 'bm' uses the default Box-Muller transformations
            method.  'zig' uses the much faster Ziggurat method of Marsaglia and Tsang.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        >>> s = np.random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311, #random
               -0.38672696, -0.4685006 ])                               #random
        >>> s.shape
        (8000,)
        >>> s = np.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        """
        key = np.dtype(dtype).name
        if key == 'float64':
            if method == u'zig':
                return double_fill(&self.rng_state, &random_gauss_zig_double_fill,
                                   size, self.lock, out)
            else:
                return double_fill(&self.rng_state, &random_gauss_fill,
                                   size, self.lock, out)
        elif key == 'float32':
            if method == u'zig':
                return float_fill(&self.rng_state, &random_gauss_zig_float_fill,
                                   size, self.lock, out)
            else:
                return float_fill(&self.rng_state, &random_gauss_fill_float,
                                   size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype "%s" for standard_normal' % key)



    def normal(self, loc=0.0, scale=1.0, size=None, method=__normal_method):
        """
        normal(loc=0.0, scale=1.0, size=None, method='bm')

        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first
        derived by De Moivre and 200 years later by both Gauss and Laplace
        independently [2]_, is often called the bell curve because of
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature.  For example, it
        describes the commonly occurring distribution of samples influenced
        by a large number of tiny, random disturbances, each with its own
        unique distribution [2]_.

        Parameters
        ----------
        loc : float or array_like of floats
            Mean ("centre") of the distribution.
        scale : float or array_like of floats
            Standard deviation (spread or "width") of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``loc`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
        method : str, optional
            Either 'bm' or 'zig'. 'bm' uses the default Box-Muller transformations
            method.  'zig' uses the much faster Ziggurat method of Marsaglia and Tsang.


        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized normal distribution.

        See Also
        --------
        scipy.stats.norm : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability density for the Gaussian distribution is

        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard
        deviation. The square of the standard deviation, :math:`\\sigma^2`,
        is called the variance.

        The function has its peak at the mean, and its "spread" increases with
        the standard deviation (the function reaches 0.607 times its maximum at
        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
        `numpy.random.normal` is more likely to return samples lying close to
        the mean, rather than those far away.

        References
        ----------
        .. [1] Wikipedia, "Normal distribution",
               http://en.wikipedia.org/wiki/Normal_distribution
        .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
               Random Variables and Random Signal Principles", 4th ed., 2001,
               pp. 51, 51, 125.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 0, 0.1 # mean and standard deviation
        >>> s = np.random.normal(mu, sigma, 1000)

        Verify the mean and the variance:

        >>> abs(mu - np.mean(s)) < 0.01
        True

        >>> abs(sigma - np.std(s, ddof=1)) < 0.01
        True

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        """
        if method == 'bm':
            return cont(&self.rng_state, &random_normal, size, self.lock, 2,
                        loc, '', CONS_NONE,
                        scale, 'scale', CONS_NON_NEGATIVE,
                        0.0, '', CONS_NONE,
                        None)
        else:
            return cont(&self.rng_state, &random_normal_zig, size, self.lock, 2,
                        loc, '', CONS_NONE,
                        scale, 'scale', CONS_NON_NEGATIVE,
                        0.0, '', CONS_NONE,
                        None)
