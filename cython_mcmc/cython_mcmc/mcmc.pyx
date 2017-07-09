# cython: profile=True

cimport cython
from srs.mt19937 cimport RandomState
from srs.mt19937 import RandomState
from libc.math cimport log as clog, pi as cpi, exp as cexp
import numpy as np
cimport numpy as cnp

DEF LOG_2_PI = 1.8378770664093453

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double norm_logpdf(const double mu, const double sigma, const double* const x, const int n):
    cdef double s = 0.0
    cdef double ONE_OVER_SIG = 1.0 / sigma
    cdef int i
    for i in range(n):
        s += (x[i] - mu) * (x[i] - mu)
    return - 0.5 * n * LOG_2_PI - n * clog(sigma) - 0.5 * ONE_OVER_SIG * ONE_OVER_SIG * s

cdef RandomState _rs = RandomState()

cdef double sample_norm(double mu, double sigma):
    return _rs.c_standard_normal() * sigma + mu

cdef bint accept_p(double log_p_accept):
    return _rs.c_random_sample() < cexp(log_p_accept)

@cython.boundscheck(False)
@cython.wraparound(False)
def log_sampler(cnp.ndarray[double] data,
                int samples,
                double mu_init=.5,
                double proposal_width=.5,
                double mu_prior_mu=0,
                double mu_prior_sd=1.):
    
    cdef:
        double mu_proposal
        double log_likelihood_current, log_likelihood_proposal
        double log_prior_current, log_prior_proposal
        double log_p_current, log_p_proposal
        double log_p_accept
        bint accept
        double mu_current = mu_init
        list posterior = [mu_current]
        double *cdata = &data[0]
        int ndata = data.shape[0]
        cnp.ndarray[double] np_buf = np.empty((1,), dtype='f8')
        double *buf1 = &np_buf[0]
        int i

    for i in range(samples):
        # suggest new position
        # mu_proposal = sample_norm(_rs, mu_current, proposal_width)
        mu_proposal = sample_norm(mu_current, proposal_width)

        # Compute likelihood by adding log probabilities of each data point
        log_likelihood_current = norm_logpdf(mu_current, 1, cdata, ndata)
        log_likelihood_proposal = norm_logpdf(mu_proposal, 1, cdata, ndata)
        
        # Compute prior log probability of current and proposed mu
        buf1[0] = mu_current
        log_prior_current = norm_logpdf(mu_prior_mu, mu_prior_sd, buf1, 1)
        buf1[0] = mu_proposal
        log_prior_proposal = norm_logpdf(mu_prior_mu, mu_prior_sd, buf1, 1)
        
        log_p_current = log_likelihood_current + log_prior_current
        log_p_proposal = log_likelihood_proposal + log_prior_proposal
        
        # Accept proposal?
        log_p_accept = log_p_proposal - log_p_current
        
        # if accept_p(_rs, log_p_accept):
        if accept_p(log_p_accept):
            mu_current = mu_proposal
        
        posterior.append(mu_current)
        
    return posterior
