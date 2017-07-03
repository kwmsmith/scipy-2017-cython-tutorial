#include <math.h>
#include <stdlib.h>

#include "Python.h"
#include "numpy/npy_common.h"

#ifdef _WIN32
#include "src/common/stdint.h"
typedef int bool;
#define false 0
#define true 1
#else
#include <stdint.h>
#endif

#ifndef min
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

#if  defined(RS_PCG32)
#include "interface/pcg-32/pcg-32-shim.h"
#elif defined(RS_PCG64)
#include "interface/pcg-64/pcg-64-shim.h"
#elif defined(RS_RANDOMKIT)
#include "interface/random-kit/random-kit-shim.h"
#elif defined(RS_XORSHIFT128)
#include "interface/xorshift128/xorshift128-shim.h"
#elif defined(RS_XORSHIFT1024)
#include "interface/xorshift1024/xorshift1024-shim.h"
#elif defined(RS_XOROSHIRO128PLUS)
#include "interface/xoroshiro128plus/xoroshiro128plus-shim.h"
#elif defined(RS_MRG32K3A)
#include "interface/mrg32k3a/mrg32k3a-shim.h"
#elif defined(RS_MLFG_1279_861)
#include "interface/mlfg-1279-861/mlfg-1279-861-shim.h"
#elif defined(RS_DSFMT)
#include "interface/dSFMT/dSFMT-shim.h"
#else
#error Unknown RNG!!!  Unknown RNG!!!  Unknown RNG!!!
#endif

extern int64_t random_positive_int64(aug_state* state);

extern int32_t random_positive_int32(aug_state* state);

extern float random_standard_uniform_float(aug_state* state);

extern double random_standard_uniform_double(aug_state* state);

extern double random_standard_exponential(aug_state* state);

extern double random_gauss(aug_state* state);

extern double random_standard_gamma(aug_state* state, double shape);

extern float random_standard_gamma_float(aug_state* state, float shape);

extern double random_normal(aug_state *state, double loc, double scale);

extern double random_normal_zig(aug_state *state, double loc, double scale);

extern double random_exponential(aug_state *state, double scale);

extern double random_uniform(aug_state *state, double loc, double scale);

extern double random_gamma(aug_state *state, double shape, double scale);

extern float random_gamma_float(aug_state *state, float shape, float scale);

extern double random_beta(aug_state *state, double a, double b);

extern double random_chisquare(aug_state *state, double df);

extern double random_f(aug_state *state, double dfnum, double dfden);

extern long random_negative_binomial(aug_state *state, double n, double p);

extern double random_standard_cauchy(aug_state *state);

extern double random_standard_t(aug_state *state, double df);

extern double random_pareto(aug_state *state, double a);

extern double random_weibull(aug_state *state, double a);

extern double random_power(aug_state *state, double a);

extern double random_laplace(aug_state *state, double loc, double scale);

extern double random_gumbel(aug_state *state, double loc, double scale);

extern double random_logistic(aug_state *state, double loc, double scale);

extern double random_lognormal(aug_state *state, double mean, double sigma);

extern double random_rayleigh(aug_state *state, double mode);

extern double random_gauss_zig(aug_state* state);

extern double random_gauss_zig_julia(aug_state* state);

extern double random_noncentral_chisquare(aug_state *state, double df, double nonc);

extern double random_noncentral_f(aug_state *state, double dfnum, double dfden, double nonc);

extern double random_wald(aug_state *state, double mean, double scale);

extern double random_vonmises(aug_state *state, double mu, double kappa);

extern double random_triangular(aug_state *state, double left, double mode, double right);

extern long random_poisson(aug_state *state, double lam);

extern long random_negative_binomial(aug_state *state, double n, double p);

extern long random_binomial(aug_state *state, double p, long n);

extern long random_logseries(aug_state *state, double p);

extern long random_geometric(aug_state *state, double p);

extern long random_zipf(aug_state *state, double a);

extern long random_hypergeometric(aug_state *state, long good, long bad, long sample);

extern long random_positive_int(aug_state* state);

extern unsigned long random_uint(aug_state* state);

extern unsigned long random_interval(aug_state* state, unsigned long max);

extern void random_bounded_uint64_fill(aug_state *state, uint64_t off, uint64_t rng, npy_intp cnt, uint64_t *out);

extern void random_bounded_uint32_fill(aug_state *state, uint32_t off, uint32_t rng, npy_intp cnt, uint32_t *out);

extern void random_bounded_uint16_fill(aug_state *state, uint16_t off, uint16_t rng, npy_intp cnt, uint16_t *out);

extern void random_bounded_uint8_fill(aug_state *state, uint8_t off, uint8_t rng, npy_intp cnt, uint8_t *out);

extern void random_bounded_bool_fill(aug_state *state, npy_bool off, npy_bool rng, npy_intp cnt, npy_bool *out);

extern void random_uniform_fill_float(aug_state* state, npy_intp count, float *out);

extern void random_uniform_fill_double(aug_state* state, npy_intp count, double *out);

extern void random_standard_exponential_fill_double(aug_state* state, npy_intp count, double *out);

extern void random_standard_exponential_fill_float(aug_state* state, npy_intp count, float *out);

extern void random_gauss_fill(aug_state* state, npy_intp count, double *out);

extern void random_gauss_fill_float(aug_state* state, npy_intp count, float *out);

extern void random_gauss_zig_julia_fill(aug_state *state, npy_intp count, double *out);

extern void random_gauss_zig_float_fill(aug_state *state, npy_intp count, float *out);

extern void random_gauss_zig_double_fill(aug_state *state, npy_intp count, double *out);