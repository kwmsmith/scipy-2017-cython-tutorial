cdef extern from "distributions.h":

    cdef struct s_binomial_t:
        int has_binomial
        double psave
        long nsave
        double r
        double q
        double fm
        long m
        double p1
        double xm
        double xl
        double xr
        double c
        double laml
        double lamr
        double p2
        double p3
        double p4

    ctypedef s_binomial_t binomial_t
