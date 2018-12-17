#pragma once

#include <iostream>
#include <chrono>
#include <cstring>
#include <ctime>

#include "binary_utils.cuh"
#include "des.cuh"
#include "des_constant_cpu.cuh"
#include "des_utils.cuh"

__host__ bool cpu_brute_force(char *key_alphabet, int key_length, char *message_alphabet,
                     int message_length, uint64_t ciphertext, uint64_t *message_result, uint64_t *key_result);
__host__ void des_brute_force_cpu(char *key_alphabet, int key_length, char *message_alphabet, int message_length, uint64_t ciphertext);


__host__ bool cpu_brute_force(char *key_alphabet, int key_length, char *message_alphabet,
                     int message_length, uint64_t ciphertext, uint64_t *message_result, uint64_t *key_result)
{
    int64_t key_alphabet_length = (int64_t)std::strlen(key_alphabet);
    int64_t message_alphabet_length = (int64_t)std::strlen(message_alphabet);
    uint64_t keys_count = get_combinations_count(key_alphabet_length, key_length);
    uint64_t messages_cout = get_combinations_count(message_alphabet_length, message_length);
    uint64_t subkeyes[16];

    for (uint64_t i = 0; i < keys_count; i++)
    {
        uint64_t key = create_combination(i, key_alphabet, key_alphabet_length, key_length);
        create_subkeyes(key, subkeyes, cpu_SHIFTS, cpu_PC_1, cpu_PC_2);

        for (uint64_t j = 0; j < messages_cout; j++)
        {
            uint64_t message = create_combination(j, message_alphabet, message_alphabet_length, message_length);
            if (ciphertext == des_encrypt(message, subkeyes, cpu_IP, cpu_IP_REV, cpu_E_BIT, cpu_P, cpu_S))
            {
                *key_result = key;
                *message_result = message;
                return true;
            }
        }
    }

    return false;
}

__host__ void des_brute_force_cpu(char *key_alphabet, int key_length, char *message_alphabet, int message_length, uint64_t ciphertext)
{
    std::cout << "\nDES CPU" << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint64_t key, message;
    bool found_key = cpu_brute_force(key_alphabet, key_length, message_alphabet, message_length, ciphertext, &message, &key);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    if (found_key)
    {
        print_hex(key, "Found key:");
        print_hex(message, "Found message:");
    }
    else
    {
        std::cout << "Key was not found" << std::endl;
    }

    std::cout << "Time elapsed:" << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;

    if(verify(key, message, ciphertext, cpu_IP, cpu_IP_REV, cpu_E_BIT, cpu_P, cpu_S, cpu_SHIFTS, cpu_PC_1, cpu_PC_2))
    {
        std::cout << "Verified OK." << std::endl;
    }
}
