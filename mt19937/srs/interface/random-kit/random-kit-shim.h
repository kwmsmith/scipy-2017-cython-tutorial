#ifdef _WIN32
#include "../../src/common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif

#include "../../src/common/binomial.h"
#include "../../src/common/entropy.h"
#include "../../src/random-kit/random-kit.h"

typedef struct s_aug_state {
    randomkit_state *rng;
    binomial_t *binomial;

    int has_gauss, has_gauss_float, shift_zig_random_int, has_uint32;
    float gauss_float;
    double gauss;
    uint32_t uinteger;
    uint64_t zig_random_int;
} aug_state;

inline uint32_t random_uint32(aug_state* state)
{
    return randomkit_random(state->rng);
}

inline uint64_t random_uint64(aug_state* state)
{
    return (((uint64_t) randomkit_random(state->rng)) << 32) | randomkit_random(state->rng);
}

inline double random_double(aug_state* state)
{
    int32_t a = random_uint32(state) >> 5, b = random_uint32(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

inline uint64_t random_raw_values(aug_state* state)
{
    return (uint64_t)random_uint32(state);
}

extern void entropy_init(aug_state* state);

extern void set_seed_by_array(aug_state* state, uint32_t *init_key, int key_length);

extern void set_seed(aug_state* state, uint32_t seed);



