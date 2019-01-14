#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include "../../../Library/ParallelThreadProfiler.cuh"
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
	RegisterTimeMarker(0); 

	const int SLEEP_TIME = 50000000;
	switch(tid % 32)
	{
		case 0:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME);
			RegisterTimeMarker(2);
			break;
		case 1:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*2);
			RegisterTimeMarker(2);
			break;
		
		.
		.
		.
		
		case 31:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*32);
			RegisterTimeMarker(2);
			break;
	}

	RegisterTimeMarker(3);
}

int main()
{
	cudaError_t cudaStatus;
	ParallelThreadProfiler::InitialiseProfiling();

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

	const int blocks = 2;
	const int thread_per_block = 32;

	ParallelThreadProfiler::CreateLabel("start",0);
	ParallelThreadProfiler::CreateLabel("switch_start",1);
	ParallelThreadProfiler::CreateLabel("switch_end",2);
	ParallelThreadProfiler::CreateLabel("end",3);
	ParallelThreadProfiler::InitialiseKernelProfiling("single_loop_kernel",blocks*thread_per_block,4);

	single_loop<<<blocks,thread_per_block>>>(dev_limits);

	ParallelThreadProfiler::SaveResults();

	cudaFree(dev_limits);
	cudaDeviceSynchronize();
	cudaStatus = cudaDeviceReset();
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"cudaDeviceReset failed!!!");
		return 1;
	}

	cout << endl <<"End" << endl;
	return 0;
}

