#include "xoroshiro128plus.h"
#include "../splitmix64/splitmix64.h"

extern inline uint64_t xoroshiro128plus_next(xoroshiro128plus_state* state);

static inline uint64_t rotl(const uint64_t x, int k);

void xoroshiro128plus_jump(xoroshiro128plus_state* state) {
	static const uint64_t JUMP[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };

    int i, b;
	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for(i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b) {
				s0 ^= state->s[0];
				s1 ^= state->s[1];
			}
			xoroshiro128plus_next(state);
		}

    state->s[0] = s0;
    state->s[1] = s1;
}

void xoroshiro128plus_seed(xoroshiro128plus_state* state, uint64_t seed)
{

    uint64_t seed_copy = seed;
    uint64_t state1 = splitmix64_next(&seed_copy);
    uint64_t state2 = splitmix64_next(&seed_copy);
    xoroshiro128plus_init_state(state, state1, state2);
}

void xoroshiro128plus_seed_by_array(xoroshiro128plus_state* state, uint64_t *seed_array, int count)
{
    uint64_t initial_state[2] = {0};
    uint64_t seed_copy = 0;
    int iter_bound = 2>=count ? 2 : count;
    int i, loc = 0;
    for (i = 0; i < iter_bound; i++)
    {
        if (i < count)
            seed_copy ^= seed_array[i];
        initial_state[loc] = splitmix64_next(&seed_copy);
        loc ++;
        if (loc == 2)
            loc = 0;
    }
    xoroshiro128plus_init_state(state, initial_state[0], initial_state[1]);
}

void xoroshiro128plus_init_state(xoroshiro128plus_state* state, uint64_t seed, uint64_t inc)
{
    state->s[0] = seed;
    state->s[1] = inc;
}

