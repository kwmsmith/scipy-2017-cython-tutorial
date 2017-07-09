#include <inttypes.h>
#include <stdio.h>
#include "xoroshiro128plus.h"

int main(void)
{
    int i;
    uint64_t temp, seed = 1ULL;
    xoroshiro128plus_state state = {{ 0 }};
    xoroshiro128plus_seed(&state, seed);

    FILE *fp;
    fp = fopen("xoroshiro128plus-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = xoroshiro128plus_next(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);

    seed = 12345678910111ULL;
    xoroshiro128plus_seed(&state, seed);
    fp = fopen("xoroshiro128plus-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu64 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = xoroshiro128plus_next(&state);
        fprintf(fp, "%d, %" PRIu64 "\n", i, temp);
        printf("%d, %" PRIu64 "\n", i, temp);
    }
    fclose(fp);
}