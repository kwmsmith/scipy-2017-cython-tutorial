#include <stdio.h>
#include <inttypes.h>
#include "randomkit.h"

int main(void)
{
    int i;
    uint32_t temp, seed = 1UL;
    randomkit_state state = {{ 0 }};
    randomkit_seed(&state, seed);

    FILE *fp;
    fp = fopen("randomkit-testset-1.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu32 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = randomkit_random(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);

    seed = 123456789UL;
    randomkit_seed(&state, seed);
    fp = fopen("randomkit-testset-2.csv", "w");
    if(fp == NULL){
         printf("Couldn't open file\n");
         return -1;
    }
    fprintf(fp, "seed, %" PRIu32 "\n", seed);
    for (i=0; i < 1000; i++)
    {
        temp = randomkit_random(&state);
        fprintf(fp, "%d, %" PRIu32 "\n", i, temp);
        printf("%d, %" PRIu32 "\n", i, temp);
    }
    fclose(fp);
}