#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include "../../../Library/CudaThreadProfiler.cuh"
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

#define WARP_SIZE 32



__device__ void sleep(clock_value_t sleep_cycles)
{
	clock_value_t start = clock64();
	clock_value_t cycles_elapsed;
	do { cycles_elapsed = clock64() - start; } while (cycles_elapsed < sleep_cycles);
}

__global__ void single_loop(int* limits)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int M = limits[threadIdx.x];
	RegisterTimeMarker(0); 

	const int SLEEP_TIME = 50000000;

	for(int i=0;i<M;i++){
		sleep(SLEEP_TIME);	
	}

	RegisterTimeMarker(1);
}

int main()
{
	cudaError_t cudaStatus;
	CudaThreadProfiler::InitialiseProfiling();

	cout << endl << "GPU computations started..." << endl;
	srand(time(NULL));
	
	int limits[WARP_SIZE] = {};
	for(int i=0;i<WARP_SIZE;i++)
	{
		limits[i]=WARP_SIZE-i;
	}

	int* dev_limits=NULL;
	cudaMalloc((void**)&dev_limits,WARP_SIZE*sizeof(int));
	cudaMemcpy(dev_limits,limits,WARP_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	checkCudaErrors(cudaPeekAtLastError());

	const int blocks = 32;
	const int thread_per_block = 32;

	CudaThreadProfiler::CreateLabel("start",0);
	CudaThreadProfiler::CreateLabel("end",1);
	CudaThreadProfiler::InitialiseKernelProfiling("single_loop_kernel",blocks*thread_per_block,2);

	single_loop<<<blocks,thread_per_block>>>(dev_limits);

	CudaThreadProfiler::SaveResults();

	cudaFree(dev_limits);
	cudaStatus = cudaDeviceReset();
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"cudaDeviceReset failed!!!");
		return 1;
	}

	cout << endl <<"End" << endl;
	return 0;
}

