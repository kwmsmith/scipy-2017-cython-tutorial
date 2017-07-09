from binomial cimport binomial_t
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t, intptr_t)
include "rk_state_len.pxi"

cdef extern from "distributions.h":

    cdef struct s_randomkit_state:
      uint32_t key[RK_STATE_LEN]
      int pos

    ctypedef s_randomkit_state randomkit_state

    cdef struct s_aug_state:
        randomkit_state *rng
        binomial_t *binomial

        int has_gauss, shift_zig_random_int, has_uint32, has_gauss_float
        float gauss_float
        double gauss
        uint64_t zig_random_int
        uint32_t uinteger

    ctypedef s_aug_state aug_state
    cdef void set_seed(aug_state* state, uint32_t seed)
    cdef void set_seed_by_array(aug_state* state, uint32_t *init_key, int key_length)

ctypedef randomkit_state rng_t

