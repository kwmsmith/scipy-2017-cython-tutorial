#include <stddef.h>
#include <math.h>
#ifdef _WIN32
#include "../common/stdint.h"
#define inline __forceinline
#else
#include <stdint.h>
#endif

typedef struct s_xoroshiro128plus_state
{
    uint64_t s[2];
} xoroshiro128plus_state;

void xoroshiro128plus_jump(xoroshiro128plus_state* state);

void xoroshiro128plus_seed(xoroshiro128plus_state* state, uint64_t seed);

void xoroshiro128plus_seed_by_array(xoroshiro128plus_state* state, uint64_t *seed_array, int count);

void xoroshiro128plus_init_state(xoroshiro128plus_state* state, uint64_t seed, uint64_t inc);

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

inline uint64_t xoroshiro128plus_next(xoroshiro128plus_state* state) {
	const uint64_t s0 = state->s[0];
	uint64_t s1 = state->s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	state->s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	state->s[1] = rotl(s1, 36); // c

	return result;
}

