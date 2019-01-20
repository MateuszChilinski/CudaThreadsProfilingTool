#include <iostream>
#include <sstream>
#include <math.h>

#include "usage.cuh"
#include "binary_utils.cuh"
#include "des_constant_cpu.cuh"
#include "des.cuh"
#include "des_utils.cuh"
#include "des_bf_cpu.cuh"
#include "des_bf_gpu.cuh"

#include "../../Library/ParallelThreadProfiler.cuh"

int main(int argc, char** argv)
{
    int key_length;
    char *key_alphabet;
    int message_length;
    char *message_alphabet;
    uint64_t ciphertext;
    bool useCpu = false;
    bool useGpu = false;

    get_parameters(argc, argv, &ciphertext, &key_alphabet, &key_length, &message_alphabet, &message_length, &useCpu, &useGpu);
    usage();

    printf("\n\nParameters: \n");
    print_hex(ciphertext, "Ciphertext: ");
    printf("Key length: %d\n", key_length);
    printf("Key alphabet: %s\n", key_alphabet);
    printf("Message length: %d\n", message_length);
    printf("Message alphabet: %s\n", message_alphabet);
    printf("Use cpu: %s\n", useCpu ? "true" : "false");
    printf("Use gpu: %s\n", useGpu ? "true" : "false");
    printf("\n");


	ParallelThreadProfiler::InitialiseProfiling();
    if (useCpu)
    {
        des_brute_force_cpu(key_alphabet, key_length, message_alphabet, message_length, ciphertext);
    }
    if (useGpu)
    {
        des_brute_force_gpu(key_alphabet, key_length, message_alphabet, message_length, ciphertext);
    }

    return 0;
}