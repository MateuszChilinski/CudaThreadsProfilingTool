#pragma once

#include <iostream>

#include "binary_utils.cuh"

__host__ __device__ uint64_t f(uint64_t right, uint64_t subkey, int *E_BIT, int *P, int **S);
__host__ __device__ void create_subkeyes(uint64_t key, uint64_t *subkeyes, int *SHIFTS, int *PC_1, int *PC_2);
__host__ __device__ uint64_t des_encrypt(uint64_t message, uint64_t *subkeyes, int *IP, int *IP_REV, int *E_BIT, int *P, int **S);
__host__ __device__ bool verify(uint64_t key, uint64_t message, uint64_t ciphertext, int *IP, int *IP_REV, int *E_BIT, int *P, int **S, int *SHIFTS, int *PC_1, int *PC_2);

__host__ __device__ void create_subkeyes(uint64_t key, uint64_t *subkeyes, int *SHIFTS, int *PC_1, int *PC_2)
{
    uint64_t k_plus = permutate(key, PC_1, 56, 64);
    uint64_t c_0, d_0;
    split_bits56(k_plus, &c_0, &d_0);

    uint64_t c_tmp, d_tmp, cd_tmp, c_prev = c_0, d_prev = d_0;

    for (int i = 1; i <= 16; i++)
    {
        c_tmp = bits_cycle_left56(c_prev, SHIFTS[i - 1]);
        d_tmp = bits_cycle_left56(d_prev, SHIFTS[i - 1]);
        c_prev = c_tmp;
        d_prev = d_tmp;

        cd_tmp = c_tmp << 28 | d_tmp;
        subkeyes[i - 1] = permutate(cd_tmp, PC_2, 48, 56);
    }
}

__host__ __device__ uint64_t f(uint64_t right, uint64_t subkey, int *E_BIT, int *P, int **S)
{
    right = permutate(right, E_BIT, 48, 32);
    right ^= subkey;

    uint64_t result = 0;

    for (int i = 0; i < 8; i++)
    {
        uint64_t left_tmp = (right << (16 + 6 * i)) >> 63;
        uint64_t right_tmp = (right << (21 + 6 * i)) >> 63;
        uint64_t outer = left_tmp << 1 | right_tmp;
        uint64_t inner = (right << (17 + 6 * i)) >> 60;
        uint64_t sbox_value = S[i][outer << 4 | inner];
        result ^= sbox_value << (28 - 4 * i);
    }

    result = permutate(result, P, 32, 32);

    return result;
}

__host__ __device__ uint64_t des_encrypt(uint64_t message, uint64_t *subkeyes, int *IP, int *IP_REV, int *E_BIT, int *P, int **S)
{
    uint64_t ip = permutate(message, IP, 64, 64);
    uint64_t left, right;
    split_bits64(ip, &left, &right);

    uint64_t left_prev, right_prev;
    for (int i = 0; i < 16; i++)
    {
        left_prev = left;
        right_prev = right;
        left = right_prev;
        right = left_prev ^ f(right_prev, subkeyes[i], E_BIT, P, S);
    }

    uint64_t rl = right << 32 | left;
    uint64_t encrypted = permutate(rl, IP_REV, 64, 64);

    return encrypted;
}

__host__ bool verify(uint64_t key, uint64_t message, uint64_t ciphertext, int *IP, int *IP_REV, int *E_BIT, int *P, int **S, int *SHIFTS, int *PC_1, int *PC_2)
{
    uint64_t subkeyes[16];
    create_subkeyes(key, subkeyes, SHIFTS, PC_1, PC_2);
    return ciphertext == des_encrypt(message, subkeyes, IP, IP_REV, E_BIT, P, S);
}