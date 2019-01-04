#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <iostream>

#include "des.h"
#include "usefull.h"
#include "cracking.h"

#include "../../Library/CudaThreadProfiler.cuh"

// to w koncu nie dziala
//https://stackoverflow.com/questions/2619296/how-to-return-a-single-variable-from-a-cuda-kernel-function
//__device__ bool d_found = false;
//__device__ uint64_t d_value = 0;

void user_input(char* start_char, int* alphabet_size, char data[], int* key_length)
{
	std::cout << "Enter key length: ";
	std::cin >> *key_length;

	std::cout << "Enter alphabet size: ";
	std::cin >> *alphabet_size;

	std::cout << "Enter alphabet start char: ";
	std::cin >> *start_char;

	std::cout << "Enter message to encrypt: ";
	scanf("%8s", data);
}

//nvcc kernel.cu -o des -gencode arch=compute_60,code=sm_60
int main()
{
	uint64_t message, key;
	//key = 0b0001001100110100010101110111100110011011101111001101111111110001;
	//message = 0b0000000100100011010001010110011110001001101010111100110111101111;
	int key_length = 8;
	int alphabet_size = 4;
	char start_char = 'a';
	char data[9] = { 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd' };

	//user_input(&start_char, &alphabet_size, data, &key_length);

	key = generate_key((1 << key_length) - 1);
	message = generate_message(start_char, data, alphabet_size);
	uint64_t cypher = des(message, key, true);

	std::cout << "Message: ";
	bin_print(message);
	std::cout << "Key:     ";
	bin_print(key);
	std::cout << "Cypher:  ";
	bin_print(cypher);

	CudaThreadProfiler::InitialiseProfiling();
	run_kernel(32, 512, cypher, key, key_length, start_char, alphabet_size, data);
    return 0;
}
