#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h" // to chyba nie jest potrzebne
#include "cuda.h"

#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s\nFILE: %s\n LINE: %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}