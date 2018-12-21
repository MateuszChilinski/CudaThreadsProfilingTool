#pragma once

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <ctime>

#include "binary_utils.cuh"
#include "des.cuh"
#include "des_constant_cpu.cuh"
#include "des_constant_gpu.cuh"
#include "des_utils.cuh"
#include "../../Library/CudaThreadProfiler.cuh"

__global__ void gpu_brute_force(char *key_alphabet, int64_t key_alphabet_length, int key_length, char *message_alphabet, int64_t message_alphabet_length,
    int message_length, uint64_t ciphertext, uint64_t *message_result, uint64_t *key_result, bool *found_key);
    
__host__ void des_brute_force_gpu(char *key_alphabet, int key_length, char *message_alphabet, int message_length, uint64_t ciphertext);


__global__ void gpu_brute_force(char *key_alphabet, int64_t key_alphabet_length, int key_length, char *message_alphabet, int64_t message_alphabet_length,
                     int message_length, uint64_t ciphertext, uint64_t *message_result, uint64_t *key_result, bool *found_key)
{
	RegisterTimeMarker("start");
    uint64_t keys_count = get_combinations_count(key_alphabet_length, key_length);
    uint64_t messages_cout = get_combinations_count(message_alphabet_length, message_length);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    uint64_t subkeyes[16];
    for (uint64_t i = index; i < keys_count && !(*found_key); i += stride)
    {
        uint64_t key = create_combination(i, key_alphabet, key_alphabet_length, key_length);
        create_subkeyes(key, subkeyes, gpu_SHIFTS, gpu_PC_1, gpu_PC_2);

        for (uint64_t j = 0; j < messages_cout && !(*found_key); j++)
        {
            uint64_t message = create_combination(j, message_alphabet, message_alphabet_length, message_length);
            uint64_t test_cipher = des_encrypt(message, subkeyes, gpu_IP, gpu_IP_REV, gpu_E_BIT, gpu_P, gpu_S);
            if (ciphertext == test_cipher)
            {
                *key_result = key;
                *message_result = message;
                *found_key = true;
                return;
            }
        }
    }

	RegisterTimeMarker("end");
}

__host__ void des_brute_force_gpu(char *key_alphabet, int key_length, char *message_alphabet, int message_length, uint64_t ciphertext)
{
    std::cout << "\nDES GPU" << std::endl;    
    uint64_t *key;
    uint64_t *message;
    bool *found_key;
    
    int64_t key_alphabet_length = (int64_t)std::strlen(key_alphabet);
    int64_t message_alphabet_length = (int64_t)std::strlen(message_alphabet);

    char *gpu_key_alphabet, *gpu_message_alphabet;
     
    cudaMallocManaged(&gpu_key_alphabet, key_alphabet_length*sizeof(char));
    cudaMallocManaged(&gpu_message_alphabet, message_alphabet_length*sizeof(char));
    cudaMallocManaged(&key, sizeof(uint64_t));
    cudaMallocManaged(&message, sizeof(uint64_t));
    cudaMallocManaged(&found_key, sizeof(bool));

    cudaMemcpy(gpu_key_alphabet, key_alphabet, key_alphabet_length, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_message_alphabet, message_alphabet, message_alphabet_length, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;
  
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    int block=4096;
    int thread =1024;

	CudaThreadProfiler::InitialiseKernelProfiling((block*thread)/32, 2);
    gpu_brute_force<<<block,thread>>>(
        gpu_key_alphabet, 
        key_alphabet_length, 
        key_length, 
        gpu_message_alphabet, 
        message_alphabet_length,
        message_length, 
        ciphertext, 
        message, 
        key, 
        found_key);

    cudaDeviceSynchronize();
	CudaThreadProfiler::SaveResults();
  
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);

    cudaEventSynchronize(stop);

    
    if (*found_key)
    {
        print_hex(*key, "Found key:");
        print_hex(*message, "Found message:");
    }
    else
    {
        std::cout << "Key was not found" << std::endl;
    }

    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Time elapsed: %f ms\n" ,elapsedTime);

    if(verify(*key, *message, ciphertext, cpu_IP, cpu_IP_REV, cpu_E_BIT, cpu_P, cpu_S, cpu_SHIFTS, cpu_PC_1, cpu_PC_2))
    {
        std::cout << "Verified OK." << std::endl;
    }
    else
    {
        std::cout << "Not verified" << std::endl;
    }

    cudaFree(key_alphabet);
    cudaFree(message_alphabet);
    cudaFree(&key);
    cudaFree(&message);
    cudaFree(&found_key);
}
