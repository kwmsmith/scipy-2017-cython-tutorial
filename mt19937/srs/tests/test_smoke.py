import pickle
import time

try:
    import cPickle
except ImportError:
    cPickle = pickle
import sys
import os
import unittest
import numpy as np
from srs import mt19937
from numpy.testing import assert_almost_equal, assert_equal, assert_raises, assert_, assert_array_equal

from nose import SkipTest


def params_0(f):
    val = f()
    assert_(np.isscalar(val))
    val = f(10)
    assert_(val.shape == (10,))
    val = f((10, 10))
    assert_(val.shape == (10, 10))
    val = f((10, 10, 10))
    assert_(val.shape == (10, 10, 10))
    val = f(size=(5, 5))
    assert_(val.shape == (5, 5))


def params_1(f, bounded=False):
    a = 5.0
    b = np.arange(2.0, 12.0)
    c = np.arange(2.0, 102.0).reshape(10, 10)
    d = np.arange(2.0, 1002.0).reshape(10, 10, 10)
    e = np.array([2.0, 3.0])
    g = np.arange(2.0, 12.0).reshape(1, 10, 1)
    if bounded:
        a = 0.5
        b = b / (1.5 * b.max())
        c = c / (1.5 * c.max())
        d = d / (1.5 * d.max())
        e = e / (1.5 * e.max())
        g = g / (1.5 * g.max())

    # Scalar
    f(a)
    # Scalar - size
    f(a, size=(10, 10))
    # 1d
    f(b)
    # 2d
    f(c)
    # 3d
    f(d)
    # 1d size
    f(b, size=10)
    # 2d - size - broadcast
    f(e, size=(10, 2))
    # 3d - size
    f(g, size=(10, 10, 10))


def comp_state(state1, state2):
    identical = True
    if isinstance(state1, dict):
        for key in state1:
            identical &= comp_state(state1[key], state2[key])
    elif type(state1) != type(state2):
        identical &= type(state1) == type(state2)
    else:
        if (isinstance(state1, (list, tuple, np.ndarray)) and isinstance(state2, (list, tuple, np.ndarray))):
            for s1, s2 in zip(state1, state2):
                identical &= comp_state(s1, s2)
        else:
            identical &= state1 == state2
    return identical


def warmup(rs, n=None):
    if n is None:
        n = 11 + np.random.randint(0, 20)
    rs.standard_normal(n, method='bm')
    rs.standard_normal(n, method='zig')
    rs.standard_normal(n, method='bm', dtype=np.float32)
    rs.standard_normal(n, method='zig', dtype=np.float32)
    rs.randint(0, 2 ** 24, n, dtype=np.uint64)
    rs.randint(0, 2 ** 48, n, dtype=np.uint64)
    rs.standard_gamma(11.0, n)
    rs.standard_gamma(11.0, n, dtype=np.float32)
    rs.random_sample(n, dtype=np.float64)
    rs.random_sample(n, dtype=np.float32)


class RNG(object):
    @classmethod
    def _extra_setup(cls):
        cls.vec_1d = np.arange(2.0, 102.0)
        cls.vec_2d = np.arange(2.0, 102.0)[None, :]
        cls.mat = np.arange(2.0, 102.0, 0.01).reshape((100, 100))
        cls.seed_error = TypeError

    def _reset_state(self):
        self.rs.set_state(self.initial_state)

    def test_init(self):
        rs = self.mod.RandomState()
        state = rs.get_state()
        rs.standard_normal(1, method='bm')
        rs.standard_normal(1, method='zig')
        rs.set_state(state)
        new_state = rs.get_state()
        assert_(comp_state(state, new_state))

    def test_advance(self):
        state = self.rs.get_state()
        if hasattr(self.rs, 'advance'):
            self.rs.advance(self.advance)
            assert_(not comp_state(state, self.rs.get_state()))
        else:
            raise SkipTest

    def test_jump(self):
        state = self.rs.get_state()
        if hasattr(self.rs, 'jump'):
            self.rs.jump()
            assert_(not comp_state(state, self.rs.get_state()))
        else:
            raise SkipTest

    def test_random_uintegers(self):
        assert_(len(self.rs.random_uintegers(10)) == 10)

    def test_random_raw(self):
        assert_(len(self.rs.random_raw(10)) == 10)
        assert_(self.rs.random_raw((10, 10)).shape == (10, 10))

    def test_uniform(self):
        r = self.rs.uniform(-1.0, 0.0, size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())

    def test_uniform_array(self):
        r = self.rs.uniform(np.array([-1.0] * 10), 0.0, size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())
        r = self.rs.uniform(np.array([-1.0] * 10), np.array([0.0] * 10), size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())
        r = self.rs.uniform(-1.0, np.array([0.0] * 10), size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())

    def test_random_sample(self):
        assert_(len(self.rs.random_sample(10)) == 10)
        params_0(self.rs.random_sample)

    def test_standard_normal_zig(self):
        assert_(len(self.rs.standard_normal(10, method='zig')) == 10)

    def test_standard_normal(self):
        assert_(len(self.rs.standard_normal(10)) == 10)
        params_0(self.rs.standard_normal)

    def test_standard_gamma(self):
        assert_(len(self.rs.standard_gamma(10, 10)) == 10)
        assert_(len(self.rs.standard_gamma(np.array([10] * 10), 10)) == 10)
        params_1(self.rs.standard_gamma)

    def test_standard_exponential(self):
        assert_(len(self.rs.standard_exponential(10)) == 10)
        params_0(self.rs.standard_exponential)

    def test_standard_cauchy(self):
        assert_(len(self.rs.standard_cauchy(10)) == 10)
        params_0(self.rs.standard_cauchy)

    def test_standard_t(self):
        assert_(len(self.rs.standard_t(10, 10)) == 10)
        params_1(self.rs.standard_t)

    def test_binomial(self):
        assert_(self.rs.binomial(10, .5) >= 0)
        assert_(self.rs.binomial(1000, .5) >= 0)

    def test_reset_state(self):
        state = self.rs.get_state()
        int_1 = self.rs.random_raw(1)
        self.rs.set_state(state)
        int_2 = self.rs.random_raw(1)
        assert_(int_1 == int_2)

    def test_seed(self):
        rs = self.mod.RandomState(*self.seed)
        rs2 = self.mod.RandomState(*self.seed)
        assert_(comp_state(rs.get_state(), rs2.get_state()))

    def test_reset_state_gauss(self):
        rs = self.mod.RandomState(*self.seed)
        rs.standard_normal()
        state = rs.get_state()
        n1 = rs.standard_normal(size=10)
        rs2 = self.mod.RandomState()
        rs2.set_state(state)
        n2 = rs2.standard_normal(size=10)
        assert_array_equal(n1, n2)

    def test_reset_state_uint32(self):
        rs = self.mod.RandomState(*self.seed)
        rs.randint(0, 2 ** 24, 120, dtype=np.uint32)
        state = rs.get_state()
        n1 = rs.randint(0, 2 ** 24, 10, dtype=np.uint32)
        rs2 = self.mod.RandomState()
        rs2.set_state(state)
        n2 = rs2.randint(0, 2 ** 24, 10, dtype=np.uint32)
        assert_array_equal(n1, n2)

    def test_reset_state_uintegers(self):
        rs = self.mod.RandomState(*self.seed)
        rs.random_uintegers(bits=32)
        state = rs.get_state()
        n1 = rs.random_uintegers(bits=32, size=10)
        rs2 = self.mod.RandomState()
        rs2.set_state(state)
        n2 = rs2.random_uintegers(bits=32, size=10)
        assert_((n1 == n2).all())

    def test_shuffle(self):
        original = np.arange(200, 0, -1)
        permuted = self.rs.permutation(original)
        assert_((original != permuted).any())

    def test_permutation(self):
        original = np.arange(200, 0, -1)
        permuted = self.rs.permutation(original)
        assert_((original != permuted).any())

    def test_tomaxint(self):
        vals = self.rs.tomaxint(size=100000)
        maxsize = 0
        if os.name == 'nt':
            maxsize = 2 ** 31 - 1
        else:
            try:
                maxsize = sys.maxint
            except:
                maxsize = sys.maxsize
        if maxsize < 2 ** 32:
            assert_((vals < sys.maxsize).all())
        else:
            assert_((vals >= 2 ** 32).any())

    def test_beta(self):
        vals = self.rs.beta(2.0, 2.0, 10)
        assert_(len(vals) == 10)
        vals = self.rs.beta(np.array([2.0] * 10), 2.0)
        assert_(len(vals) == 10)
        vals = self.rs.beta(2.0, np.array([2.0] * 10))
        assert_(len(vals) == 10)
        vals = self.rs.beta(np.array([2.0] * 10), np.array([2.0] * 10))
        assert_(len(vals) == 10)
        vals = self.rs.beta(np.array([2.0] * 10), np.array([[2.0]] * 10))
        assert_(vals.shape == (10, 10))

    def test_bytes(self):
        vals = self.rs.bytes(10)
        assert_(len(vals) == 10)

    def test_chisquare(self):
        vals = self.rs.chisquare(2.0, 10)
        assert_(len(vals) == 10)
        params_1(self.rs.chisquare)
    
    def test_complex_normal(self):
        vals = self.rs.complex_normal(2.0+7.0j, 10.0, 5.0-5.0j, size=10)
        assert_(len(vals) == 10)

    def test_exponential(self):
        vals = self.rs.exponential(2.0, 10)
        assert_(len(vals) == 10)
        params_1(self.rs.exponential)

    def test_f(self):
        vals = self.rs.f(3, 1000, 10)
        assert_(len(vals) == 10)

    def test_gamma(self):
        vals = self.rs.gamma(3, 2, 10)
        assert_(len(vals) == 10)

    def test_geometric(self):
        vals = self.rs.geometric(0.5, 10)
        assert_(len(vals) == 10)
        params_1(self.rs.exponential, bounded=True)

    def test_gumbel(self):
        vals = self.rs.gumbel(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_laplace(self):
        vals = self.rs.laplace(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_logitic(self):
        vals = self.rs.logistic(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_logseries(self):
        vals = self.rs.logseries(0.5, 10)
        assert_(len(vals) == 10)

    def test_negative_binomial(self):
        vals = self.rs.negative_binomial(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_rand(self):
        state = self.rs.get_state()
        vals = self.rs.rand(10, 10, 10)
        self.rs.set_state(state)
        assert_((vals == self.rs.random_sample((10, 10, 10))).all())
        assert_(vals.shape == (10, 10, 10))
        vals = self.rs.rand(10, 10, 10, dtype=np.float32)
        assert_(vals.shape == (10, 10, 10))

    def test_randn(self):
        state = self.rs.get_state()
        vals = self.rs.randn(10, 10, 10)
        self.rs.set_state(state)
        assert_equal(vals, self.rs.standard_normal((10, 10, 10)))
        assert_equal(vals.shape, (10, 10, 10))

        state = self.rs.get_state()
        vals = self.rs.randn(10, 10, 10, method='bm')
        self.rs.set_state(state)
        assert_equal(vals, self.rs.standard_normal((10, 10, 10), method='bm'))

        state = self.rs.get_state()
        vals_inv = self.rs.randn(10, 10, 10, method='bm')
        self.rs.set_state(state)
        vals_zig = self.rs.randn(10, 10, 10, method='zig')
        assert_((vals_zig != vals_inv).any())

        vals = self.rs.randn(10, 10, 10, dtype=np.float32)
        assert_(vals.shape == (10, 10, 10))

    def test_noncentral_chisquare(self):
        vals = self.rs.noncentral_chisquare(10, 2, 10)
        assert_(len(vals) == 10)

    def test_noncentral_f(self):
        vals = self.rs.noncentral_f(3, 1000, 2, 10)
        assert_(len(vals) == 10)
        vals = self.rs.noncentral_f(np.array([3] * 10), 1000, 2)
        assert_(len(vals) == 10)
        vals = self.rs.noncentral_f(3, np.array([1000] * 10), 2)
        assert_(len(vals) == 10)
        vals = self.rs.noncentral_f(3, 1000, np.array([2] * 10))
        assert_(len(vals) == 10)

    def test_normal(self):
        vals = self.rs.normal(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_pareto(self):
        vals = self.rs.pareto(3.0, 10)
        assert_(len(vals) == 10)

    def test_poisson(self):
        vals = self.rs.poisson(10, 10)
        assert_(len(vals) == 10)
        vals = self.rs.poisson(np.array([10] * 10))
        assert_(len(vals) == 10)
        params_1(self.rs.poisson)

    def test_poisson_lam_max(self):
        vals = self.rs.poisson_lam_max
        assert_almost_equal(vals, np.iinfo('l').max - np.sqrt(np.iinfo('l').max) * 10)

    def test_power(self):
        vals = self.rs.power(0.2, 10)
        assert_(len(vals) == 10)

    def test_randint(self):
        vals = self.rs.randint(10, 20, 10)
        assert_(len(vals) == 10)

    def test_random_integers(self):
        vals = self.rs.random_integers(10, 20, 10)
        assert_(len(vals) == 10)

    def test_rayleigh(self):
        vals = self.rs.rayleigh(0.2, 10)
        assert_(len(vals) == 10)
        params_1(self.rs.rayleigh, bounded=True)

    def test_vonmises(self):
        vals = self.rs.vonmises(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_wald(self):
        vals = self.rs.wald(1.0, 1.0, 10)
        assert_(len(vals) == 10)

    def test_weibull(self):
        vals = self.rs.weibull(1.0, 10)
        assert_(len(vals) == 10)

    def test_zipf(self):
        vals = self.rs.zipf(10, 10)
        assert_(len(vals) == 10)
        vals = self.rs.zipf(self.vec_1d)
        assert_(len(vals) == 100)
        vals = self.rs.zipf(self.vec_2d)
        assert_(vals.shape == (1, 100))
        vals = self.rs.zipf(self.mat)
        assert_(vals.shape == (100, 100))

    def test_hypergeometric(self):
        vals = self.rs.hypergeometric(25, 25, 20)
        assert_(np.isscalar(vals))
        vals = self.rs.hypergeometric(np.array([25] * 10), 25, 20)
        assert_(vals.shape == (10,))

    def test_triangular(self):
        vals = self.rs.triangular(-5, 0, 5)
        assert_(np.isscalar(vals))
        vals = self.rs.triangular(-5, np.array([0] * 10), 5)
        assert_(vals.shape == (10,))

    def test_multivariate_normal(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # diagonal covariance
        x = self.rs.multivariate_normal(mean, cov, 5000)
        assert_(x.shape == (5000, 2))
        x_zig = self.rs.multivariate_normal(mean, cov, 5000, method='zig')
        assert_(x.shape == (5000, 2))
        x_inv = self.rs.multivariate_normal(mean, cov, 5000, method='bm')
        assert_(x.shape == (5000, 2))
        assert_((x_zig != x_inv).any())

    def test_multinomial(self):
        vals = self.rs.multinomial(100, [1.0 / 3, 2.0 / 3])
        assert_(vals.shape == (2,))
        vals = self.rs.multinomial(100, [1.0 / 3, 2.0 / 3], size=10)
        assert_(vals.shape == (10, 2))

    def test_dirichlet(self):
        s = self.rs.dirichlet((10, 5, 3), 20)
        assert_(s.shape == (20, 3))

    def _test_pickle(self):
        pick = pickle.dumps(self.rs)
        unpick = pickle.loads(pick)
        assert_((type(self.rs) == type(unpick)))
        assert_(comp_state(self.rs.get_state(), unpick.get_state()))

        pick = cPickle.dumps(self.rs)
        unpick = cPickle.loads(pick)
        assert_((type(self.rs) == type(unpick)))
        assert_(comp_state(self.rs.get_state(), unpick.get_state()))

    def test_version(self):
        state = self.rs.get_state()
        assert_('version' in state)
        assert_(state['version'] == 0)

    def test_seed_array(self):
        if self.seed_vector_bits is None:
            raise SkipTest

        if self.seed_vector_bits == 32:
            dtype = np.uint32
        else:
            dtype = np.uint64
        seed = np.array([1], dtype=dtype)
        self.rs.seed(seed)
        state1 = self.rs.get_state()
        self.rs.seed(1)
        state2 = self.rs.get_state()
        assert_(comp_state(state1, state2))

        seed = np.arange(4, dtype=dtype)
        self.rs.seed(seed)
        state1 = self.rs.get_state()
        self.rs.seed(seed[0])
        state2 = self.rs.get_state()
        assert_(not comp_state(state1, state2))

        seed = np.arange(1500, dtype=dtype)
        self.rs.seed(seed)
        state1 = self.rs.get_state()
        self.rs.seed(seed[0])
        state2 = self.rs.get_state()
        assert_(not comp_state(state1, state2))

        seed = 2 ** np.mod(np.arange(1500, dtype=dtype), self.seed_vector_bits - 1) + 1
        self.rs.seed(seed)
        state1 = self.rs.get_state()
        self.rs.seed(seed[0])
        state2 = self.rs.get_state()
        assert_(not comp_state(state1, state2))

    def test_seed_array_error(self):
        if self.seed_vector_bits == 32:
            out_of_bounds = 2 ** 32
        else:
            out_of_bounds = 2 ** 64

        seed = -1
        assert_raises(ValueError, self.rs.seed, seed)

        seed = np.array([-1], dtype=np.int32)
        assert_raises(ValueError, self.rs.seed, seed)

        seed = np.array([1, 2, 3, -5], dtype=np.int32)
        assert_raises(ValueError, self.rs.seed, seed)

        seed = np.array([1, 2, 3, out_of_bounds])
        assert_raises(ValueError, self.rs.seed, seed)

    def test_uniform_float(self):
        rs = self.mod.RandomState(12345)
        warmup(rs)
        state = rs.get_state()
        r1 = rs.random_sample(11, dtype=np.float32)
        rs2 = self.mod.RandomState()
        warmup(rs2)
        rs2.set_state(state)
        r2 = rs2.random_sample(11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rs.get_state(), rs2.get_state()))

    def test_gamma_floats(self):
        rs = self.mod.RandomState()
        warmup(rs)
        state = rs.get_state()
        r1 = rs.standard_gamma(4.0, 11, dtype=np.float32)
        rs2 = self.mod.RandomState()
        warmup(rs2)
        rs2.set_state(state)
        r2 = rs2.standard_gamma(4.0, 11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rs.get_state(), rs2.get_state()))

    def test_normal_floats(self):
        rs = self.mod.RandomState()
        warmup(rs)
        state = rs.get_state()
        r1 = rs.standard_normal(11, method='bm', dtype=np.float32)
        rs2 = self.mod.RandomState()
        warmup(rs2)
        rs2.set_state(state)
        r2 = rs2.standard_normal(11, method='bm', dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rs.get_state(), rs2.get_state()))

    def test_normal_zig_floats(self):
        rs = self.mod.RandomState()
        warmup(rs)
        state = rs.get_state()
        r1 = rs.standard_normal(11, method='zig', dtype=np.float32)
        rs2 = self.mod.RandomState()
        warmup(rs2)
        rs2.set_state(state)
        r2 = rs2.standard_normal(11, method='zig', dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rs.get_state(), rs2.get_state()))

    def test_output_fill(self):
        rs = self.rs
        state = rs.get_state()
        size = (31, 7, 97)
        existing = np.empty(size)
        rs.set_state(state)
        rs.standard_normal(out=existing)
        rs.set_state(state)
        direct = rs.standard_normal(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rs.set_state(state)
        rs.standard_normal(out=existing, dtype=np.float32)
        rs.set_state(state)
        direct = rs.standard_normal(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_uniform(self):
        rs = self.rs
        state = rs.get_state()
        size = (31, 7, 97)
        existing = np.empty(size)
        rs.set_state(state)
        rs.random_sample(out=existing)
        rs.set_state(state)
        direct = rs.random_sample(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rs.set_state(state)
        rs.random_sample(out=existing, dtype=np.float32)
        rs.set_state(state)
        direct = rs.random_sample(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_exponential(self):
        rs = self.rs
        state = rs.get_state()
        size = (31, 7, 97)
        existing = np.empty(size)
        rs.set_state(state)
        rs.standard_exponential(out=existing)
        rs.set_state(state)
        direct = rs.standard_exponential(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rs.set_state(state)
        rs.standard_exponential(out=existing, dtype=np.float32)
        rs.set_state(state)
        direct = rs.standard_exponential(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_gamma(self):
        rs = self.rs
        state = rs.get_state()
        size = (31, 7, 97)
        existing = np.zeros(size)
        rs.set_state(state)
        rs.standard_gamma(1.0, out=existing)
        rs.set_state(state)
        direct = rs.standard_gamma(1.0, size=size)
        assert_equal(direct, existing)

        existing = np.zeros(size, dtype=np.float32)
        rs.set_state(state)
        rs.standard_gamma(1.0, out=existing, dtype=np.float32)
        rs.set_state(state)
        direct = rs.standard_gamma(1.0, size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_gamma_broadcast(self):
        rs = self.rs
        state = rs.get_state()
        size = (31, 7, 97)
        mu = np.arange(97.0) + 1.0
        existing = np.zeros(size)
        rs.set_state(state)
        rs.standard_gamma(mu, out=existing)
        rs.set_state(state)
        direct = rs.standard_gamma(mu, size=size)
        assert_equal(direct, existing)

        existing = np.zeros(size, dtype=np.float32)
        rs.set_state(state)
        rs.standard_gamma(mu, out=existing, dtype=np.float32)
        rs.set_state(state)
        direct = rs.standard_gamma(mu, size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_fill_error(self):
        rs = self.rs
        size = (31, 7, 97)
        existing = np.empty(size)
        assert_raises(TypeError, rs.standard_normal, out=existing, dtype=np.float32)
        assert_raises(ValueError, rs.standard_normal, out=existing[::3])
        existing = np.empty(size, dtype=np.float32)
        assert_raises(TypeError, rs.standard_normal, out=existing, dtype=np.float64)

        existing = np.zeros(size, dtype=np.float32)
        assert_raises(TypeError, rs.standard_gamma, 1.0, out=existing, dtype=np.float64)
        assert_raises(ValueError, rs.standard_gamma, 1.0, out=existing[::3], dtype=np.float32)
        existing = np.zeros(size, dtype=np.float64)
        assert_raises(TypeError, rs.standard_gamma, 1.0, out=existing, dtype=np.float32)
        assert_raises(ValueError, rs.standard_gamma, 1.0, out=existing[::3])


class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        cls.mod = mt19937
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rs = cls.mod.RandomState(*cls.seed)
        cls.initial_state = cls.rs.get_state()
        cls.seed_vector_bits = 32
        cls._extra_setup()
        cls.seed_error = ValueError

    def test_numpy_state(self):
        nprs = np.random.RandomState()
        nprs.standard_normal(99)
        state = nprs.get_state()
        self.rs.set_state(state)
        state2 = self.rs.get_state()
        assert_((state[1] == state2['state'][0]).all())
        assert_((state[2] == state2['state'][1]))
        assert_((state[3] == state2['gauss']['has_gauss']))
        assert_((state[4] == state2['gauss']['gauss']))
