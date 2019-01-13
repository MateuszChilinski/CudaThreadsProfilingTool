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
		case 2:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*3);
			RegisterTimeMarker(2);
			break;
		case 3:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*4);
			RegisterTimeMarker(2);
			break;
		case 4:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*5);
			RegisterTimeMarker(2);
			break;
		case 5:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*6);
			RegisterTimeMarker(2);
			break;
		case 6:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*7);
			RegisterTimeMarker(2);
			break;
		case 7:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*8);
			RegisterTimeMarker(2);
			break;
		case 8:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*9);
			RegisterTimeMarker(2);
			break;
		case 9:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*10);
			RegisterTimeMarker(2);
			break;
		case 10:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*11);
			RegisterTimeMarker(2);
			break;
		case 11:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*12);
			RegisterTimeMarker(2);
			break;
		case 12:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*13);
			RegisterTimeMarker(2);
			break;
		case 13:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*14);
			RegisterTimeMarker(2);
			break;
		case 14:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*15);
			RegisterTimeMarker(2);
			break;
		case 15:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*16);
			RegisterTimeMarker(2);
			break;
		case 16:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*17);
			RegisterTimeMarker(2);
			break;
		case 17:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*18);
			RegisterTimeMarker(2);
			break;
		case 18:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*19);
			RegisterTimeMarker(2);
			break;
		case 19:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*20);
			RegisterTimeMarker(2);
			break;
		case 20:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*21);
			RegisterTimeMarker(2);
			break;
		case 21:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*22);
			RegisterTimeMarker(2);
			break;
		case 22:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*23);
			RegisterTimeMarker(2);
			break;
		case 23:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*24);
			RegisterTimeMarker(2);
			break;
		case 24:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*25);
			RegisterTimeMarker(2);
			break;
		case 25:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*26);
			RegisterTimeMarker(2);
			break;
		case 26:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*27);
			RegisterTimeMarker(2);
			break;
		case 27:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*28);
			RegisterTimeMarker(2);
			break;
		case 28:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*29);
			RegisterTimeMarker(2);
			break;
		case 29:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*30);
			RegisterTimeMarker(2);
			break;
		case 30:
			RegisterTimeMarker(1);
			sleep(SLEEP_TIME*31);
			RegisterTimeMarker(2);
			break;
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

	const int blocks = 2;
	const int thread_per_block = 32;

	CudaThreadProfiler::CreateLabel("start",0);
	CudaThreadProfiler::CreateLabel("switch_start",1);
	CudaThreadProfiler::CreateLabel("switch_end",2);
	CudaThreadProfiler::CreateLabel("end",3);
	CudaThreadProfiler::InitialiseKernelProfiling("single_loop_kernel",blocks*thread_per_block,4);

	single_loop<<<blocks,thread_per_block>>>(dev_limits);

	CudaThreadProfiler::SaveResults();

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

