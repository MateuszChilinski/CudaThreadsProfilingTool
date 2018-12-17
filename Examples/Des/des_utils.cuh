#pragma once

#include <iostream>
#include <math.h>

#include "binary_utils.cuh"
#include "des.cuh"
#include "des_constant_cpu.cuh"

__host__ __device__ uint64_t create_combination(uint64_t combination_number, char* alphabet, int32_t alphabet_length, int32_t combination_length);
__host__ __device__ uint64_t get_combinations_count(int alphabet_length, int input_length);
__host__ void prepare_data();

__host__ __device__ uint64_t create_combination(uint64_t combination_number, char* alphabet, int32_t alphabet_length, int32_t combination_length)
{
	uint64_t result = 0;
	for (int i = 8; --i >= 8 - combination_length;)
	{
		uint64_t y = combination_number / alphabet_length;
		result *= (1ULL << 8);
		result += alphabet[combination_number - y * alphabet_length];
		combination_number = y;
    }
    
	return result;
}

__host__ __device__ uint64_t get_combinations_count(int alphabet_length, int input_length)
{
    uint64_t result = 1;

    while(input_length--)
    {
		result *= alphabet_length;
    }
    
    return result;
}