#pragma once
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "../Common/helper_functions.h"
#include "../Common/helper_cuda.h"
#include <time.h>
#include <fstream>
#include <math.h>

#define enableProfiler 1
#define enableProfiler2 0

using namespace std; 
__global__ struct timestamp {
	int tid;
	long long int time;
	char label[20];
};

__constant__ timestamp* tst;

__device__ int base = 0;

__device__ char * my_strcpy(char *dest, const char *src) {
#if enableProfiler 1
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[i++] != 0);
	return dest;
#endif
}
__global__ void clearBase() {
#if enableProfiler 1
	base = 0;
#endif
}
__device__ void RegisterTimeMarker(char* string)
{
#if enableProfiler 1
	const unsigned long long int blockId = blockIdx.x //1D
		+ blockIdx.y * gridDim.x //2D
		+ gridDim.x * gridDim.y * blockIdx.z; //3D
	const unsigned long long int threadId = blockId * blockDim.x + threadIdx.x;
	if (((threadId >> 5) << 5) == threadId) // check if divisible by 32 ( % 32 == 0)
	{
		int idx = threadId >> 5; // divide by 32
		int mylocation = atomicAdd(&base, 1);
		timestamp tsm;
		tsm.tid = idx;
		tsm.time = clock64();
		my_strcpy(tsm.label, string);
		tst[mylocation] = tsm;
		//printf("I am thread %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d and the time is %lld\n", idx, __mysmid(), __mywarpid(), __mylaneid(), clock64());
	}
#endif
	//printf("I am thread %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}

class CudaThreadProfiler
{
	static ofstream outfile;
	static timestamp* myTst;
	static int warps;
	static int registers;
public:
	static void InitialiseProfiling();
	static void InitialiseKernelProfiling(int, int);
	static void SaveResults();
};

timestamp* CudaThreadProfiler::myTst;
ofstream CudaThreadProfiler::outfile;

int CudaThreadProfiler::warps = 0;
int CudaThreadProfiler::registers = 0;

void CudaThreadProfiler::InitialiseProfiling()
{
#if enableProfiler 1
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", timeinfo);
	std::string str(buffer);
	outfile.open("prof"+ str +".csv", std::ios_base::app);
#endif
}
static int i = 0;
void CudaThreadProfiler::InitialiseKernelProfiling(int warps_number, int registers_number = 1)
{
#if enableProfiler 1
	checkCudaErrors(cudaPeekAtLastError());
	warps = warps_number;
	registers = registers_number;
	checkCudaErrors(cudaMalloc((void **)&myTst, registers * warps * sizeof(timestamp)));
	std::cout << i++ << "\n";
	checkCudaErrors(cudaMemcpyToSymbol(tst, &myTst, sizeof(myTst)));
#endif
}
void CudaThreadProfiler::SaveResults()
{
#if enableProfiler 1
	timestamp* host_tst = new timestamp[warps*registers];
	cudaMemcpy(host_tst, myTst, sizeof(timestamp) * registers * warps, cudaMemcpyDeviceToHost);

	int device = 0;
	int clk = 1;
	cudaError_t err = cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, device);
	for (int i = 0; i < warps*registers; i++)
	{
		timestamp tstmp = host_tst[i];
		if (tstmp.tid == 0)
			continue;
		outfile << tstmp.tid << "," << tstmp.time/clk << "," << tstmp.label <<"\n";
		//outfile << tstmp.tid << "," << tstmp.time / prop->clockRate << "," << tstmp.label << "\n";
	}
	clearBase<<<1,1>>>();
	checkCudaErrors(cudaFree(myTst));
#endif
}
