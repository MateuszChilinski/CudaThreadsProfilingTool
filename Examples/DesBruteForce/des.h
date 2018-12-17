#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>

#include "constants.h"

__device__ __host__
void bin_print(uint64_t value)
{
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			printf("%d", (int)((value >> 63 - i * 8 - j) & 1));
		}
		printf(" ");
	}
	printf("\n");
}

__device__ __host__
uint64_t key_PC1(uint64_t key)
{
	uint64_t result = 0;

	for (int i = 0; i < 56; i++)
	{
		result = result << 1;
		result = result | ((key >> (64 - PC1[i])) & 1);
	}

	return result;
}

__device__ __host__
uint64_t key_PC2(uint64_t key)
{
	uint64_t result = 0;

	for (int i = 0; i < 48; i++)
	{
		result = result << 1;
		result = result | ((key >> (56 - PC2[i])) & 1);
	}

	return result;
}

__device__ __host__
void generate_subkeys(uint64_t key, uint64_t sub_keys[16])
{
	key = key_PC1(key);

	uint32_t C0, D0;

	C0 = ((key >> 28) & 0x0fffffff);
	D0 = key & 0x0fffffff;

	for (int i = 0; i < 16; i++)
	{
		uint32_t C, D;

		// To jest stare rozwi¹zanie ale ono ju¿ dzia³a
		/*
		C = (C0 << 1) | (C0 >> 27 & 0b0001);
		D = (D0 << 1) | (D0 >> 27 & 0b0001);

		if (SHIFT[i] == 2)
		{
			C = (C << 1) | (C0 >> 26 & 0b0001);
			D = (D << 1) | (D0 >> 26 & 0b0001);
		}

		C = C & 0x0fffffff;
		D = D & 0x0fffffff;
		*/

		C = C0 << SHIFT[i];
		D = D0 << SHIFT[i];

		C = ((C | (C >> 28 & 0b11)) & 0x0fffffff);
		D = ((D | (D >> 28 & 0b11)) & 0x0fffffff);

		key = (((uint64_t)C) << 28) | ((uint64_t)D);
		sub_keys[i] = key_PC2(key);

		C0 = C;
		D0 = D;
	}
}

__device__ __host__
uint64_t message_IP(uint64_t message)
{
	uint64_t result = 0;

	for (int i = 0; i < 64; i++)
	{
		result = result << 1;
		result = result | ((message >> (64 - IP[i])) & 1);
	}

	return result;
}

__device__ __host__
uint64_t message_INVERSED_IP(uint64_t message)
{
	uint64_t result = 0;

	for (int i = 0; i < 64; i++)
	{
		result = result << 1;
		result = result | ((message >> (64 - INVERSED_IP[i])) & 1);
	}

	return result;
}

__device__ __host__
uint64_t expand(uint32_t R)
{
	uint64_t result = 0;

	for (int i = 0; i < 48; i++)
	{
		result = result << 1;
		result = result | ((R >> (32 - EXPANSION[i])) & 1);
	}

	return result;
}

__device__ __host__
uint32_t permutation(uint32_t R)
{
	uint32_t result = 0;

	for (int i = 0; i < 32; i++)
	{
		result = result << 1;
		result = result | ((R >> (32 - PBOX[i])) & 1);
	}

	return result;
}

__device__ __host__
uint32_t f(uint32_t R, uint64_t key)
{
	uint64_t temp = key ^ expand(R);
	uint32_t result = 0;

	for (int i = 0; i < 8; i++)
	{
		uint16_t i_index = ((temp >> (42 - i * 6)) & 0b000001) + ((temp >> (42 - i * 6 + 4)) & 0b000010);
		uint16_t j_index = ((temp >> (42 - i * 6 + 1)) & 0b1111);

		result = result << 4;
		result = result | (SBOX[i][i_index * 16 + j_index] & 0x000f);
	}

	return permutation(result);
}

__device__ __host__
uint64_t rounds(uint64_t message, uint64_t sub_keys[], bool encrypt)
{
	message = message_IP(message);

	uint32_t L0, R0;

	L0 = (uint32_t)((message >> 32) & 0xffffffff);
	R0 = (uint32_t)(message & 0xffffffff);

	for (int i = 0; i < 16; i++)
	{
		uint32_t L, R;

		L = R0;

		if (encrypt)
		{
			R = L0 ^ f(R0, sub_keys[i]);
		}
		else
		{
			R = L0 ^ f(R0, sub_keys[15 - i]);
		}

		L0 = L;
		R0 = R;
	}

	message = ((uint64_t)R0 << 32) | ((uint64_t)L0);

	return message_INVERSED_IP(message);
}

__device__ __host__
uint64_t des(uint64_t data, uint64_t key, bool encrypt)
{
	uint64_t sub_keys[16];
	generate_subkeys(key, sub_keys);
	return rounds(data, sub_keys, encrypt);
}