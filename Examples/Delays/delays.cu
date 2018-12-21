#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include "../../Library/CudaThreadProfiler.cuh"
#pragma once
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <time.h>
#include <fstream>
#include <math.h>
using namespace std;
using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles)
{
	clock_value_t start = clock64();
	clock_value_t cycles_elapsed;
	do { cycles_elapsed = clock64() - start; } while (cycles_elapsed < sleep_cycles);
}
__global__ void GPUDelays(long long int* global_now)
{
	RegisterTimeMarker("start"); 
	long long int start = clock64();
	long long int now;
	sleep(50000000);
	RegisterTimeMarker("end");
}
int main()
{
	cudaError_t cudaStatus;
	CudaThreadProfiler::InitialiseProfiling();
	cout << endl << "GPU computations started..." << endl;
	srand(time(NULL));
	long long int *rd;
	cudaMalloc((void **)&rd, sizeof(long long int));

	CudaThreadProfiler::InitialiseKernelProfiling(100 * 100, 2);
	GPUDelays << < 32, 512>> > (rd);
	CudaThreadProfiler::SaveResults();

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

