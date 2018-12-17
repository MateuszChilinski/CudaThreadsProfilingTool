#pragma once

#include <iostream>

__device__ __host__ uint64_t get_bit(uint64_t value, int nr);
__device__ __host__ void set_bit(uint64_t *result, int nr, uint64_t value);
__device__ __host__ uint64_t permutate(uint64_t val, int *matrix, int length, int val_length);
__device__ __host__ uint64_t bits_cycle_left56(uint64_t val, const int rotations);
__device__ __host__ void split_bits56(uint64_t value, uint64_t *left, uint64_t *right);
__device__ __host__ void split_bits64(uint64_t value, uint64_t *left, uint64_t *right);

__device__ __host__ void print_bits(uint64_t value, int group_size, int length, std::string prefix);
__device__ __host__ void print_string_hex(char *value, uint64_t length, std::string prefix);
__device__ __host__ void print_hex(uint64_t value, std::string prefix);

__device__ __host__ uint64_t get_bit(uint64_t value, int nr)
{
    return (value >> nr) & 1;
}

__device__ __host__ void set_bit(uint64_t *result, int nr, uint64_t value)
{
    *result |= (value << nr);
}

__device__ __host__ uint64_t permutate(uint64_t val, int *matrix, int length, int val_length)
{
    uint64_t result = 0;
    for (int i = length - 1; i >= 0; i--)
    {
        uint64_t bit = get_bit(val, val_length - matrix[i]);
        set_bit(&result, length - 1 - i, bit);
    }
    return result;
}

__device__ __host__ uint64_t bits_cycle_left56(uint64_t val, const int rotations)
{
    uint64_t result = val;
    uint64_t left, right;
    result <<= rotations;
    split_bits56(result, &left, &right);
    result |= left;
    result = (result << 36) >> 36;
    return result;
}

__device__ __host__ void split_bits56(uint64_t value, uint64_t *left, uint64_t *right)
{
    *left = value >> 28;
    *right = (value << 36) >> 36;
}

__device__ __host__ void split_bits64(uint64_t value, uint64_t *left, uint64_t *right)
{
    *left = value >> 32;
    *right = (value << 32) >> 32;
}

__device__ __host__ void print_hex(uint64_t value, std::string prefix)
{
    //std::cout << prefix << std::hex << value << std::endl; 
}

__device__ __host__ void print_string_hex(char *value, uint64_t length, std::string prefix)
{
    //std::cout << prefix << std::endl;

    for(int i = 0; i < length; i++)
    {
        //std::cout << std::hex << (int)value[i];
    }

    //std::cout << "\n";
}

__device__ __host__ void print_bits(uint64_t value, int group_size, int length, std::string prefix)
{
    //std::cout << prefix << std::endl;
    for (int i = length - 1; i >= 0; i--)
    {
        //std::cout << get_bit(value, i);
        if (i % group_size == 0)
        {
            //std::cout << " ";
        }
    }
    //std::cout << "\n";
}