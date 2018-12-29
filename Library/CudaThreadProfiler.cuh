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

#define ENABLE_PROFILER

using namespace std; 
__global__ struct timestamp {
	int tid;
	int x;
	int y;
	int z;
	unsigned long long int time;
	char label;
};

__constant__ timestamp* tst;

__device__ unsigned long long int base = 0;

__global__ void clearBase() {
#ifdef ENABLE_PROFILER 1
	base = 0;
#endif
}
__device__ void RegisterTimeMarker(char label)
{
#ifdef ENABLE_PROFILER
	unsigned long long int mylocation = atomicAdd(&base, 1);

	tst[mylocation].x = blockIdx.x * blockDim.x + threadIdx.x;
	tst[mylocation].y = blockIdx.y * blockDim.y + threadIdx.y;
	tst[mylocation].z = blockIdx.z * blockDim.z + threadIdx.z;
	tst[mylocation].time = clock64();
	tst[mylocation].label = label;
		//printf("I am thread %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d and the time is %lld\n", idx, __mysmid(), __mywarpid(), __mylaneid(), clock64());
#endif
	//printf("I am thread %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}

class CudaThreadProfiler
{
	static ofstream outfile;
	static timestamp* myTst;
	static int threads;
	static int registers;
	static string kernelName;
	static unsigned long long firstKernelStart;
	static bool firstKernelStartCaught;
	static int savedResultNumber;
public:
	static void InitialiseProfiling();
	static void InitialiseKernelProfiling(string, unsigned long long int, int);
	static void CreateLabel(string, char);
	static void SaveResults();
};

timestamp* CudaThreadProfiler::myTst;
ofstream CudaThreadProfiler::outfile;

int CudaThreadProfiler::threads = 0;
int CudaThreadProfiler::registers = 0;
string CudaThreadProfiler::kernelName;
bool CudaThreadProfiler::firstKernelStartCaught;
unsigned long long CudaThreadProfiler::firstKernelStart;
int CudaThreadProfiler::savedResultNumber = 0;
string labels[256];

void CudaThreadProfiler::CreateLabel(string label, char number)
{
#ifdef ENABLE_PROFILER
	labels[number] = label;
#endif
}

void CudaThreadProfiler::InitialiseProfiling()
{
#ifdef ENABLE_PROFILER
	time_t rawtime;
	firstKernelStartCaught = false;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", timeinfo);
	std::string str(buffer);
	outfile.open("prof"+ str +".csv", std::ios_base::app);
	outfile << "x" << "," << "y" << "," << "z" << "," << "time" << "," << "label" << "\n";
#endif
}
static int i = 0;
void CudaThreadProfiler::InitialiseKernelProfiling(string kernel_name, unsigned long long int threads_number, int registers_number = 1)
{
#ifdef ENABLE_PROFILER
	checkCudaErrors(cudaPeekAtLastError());
	threads = threads_number;
	registers = registers_number;
	kernelName = kernel_name;
	checkCudaErrors(cudaMalloc((void **)&myTst, registers * threads * 32 * sizeof(timestamp)));
	checkCudaErrors(cudaMemcpyToSymbol(tst, &myTst, sizeof(myTst)));
#endif
}
void CudaThreadProfiler::SaveResults()
{
#ifdef ENABLE_PROFILER
	timestamp* host_tst = new timestamp[threads*registers];
	cudaMemcpy(host_tst, myTst, sizeof(timestamp) * registers * threads, cudaMemcpyDeviceToHost);

	int device = 0;
	int clk = 1;
	cudaError_t err = cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, device);
	unsigned long long int kernelStart = -1;
	unsigned long long int kernelEnd = 0;
	for (int i = 0; i < threads*registers; i++)
	{
		unsigned long long int time = host_tst[i].time/clk;
		if (time > kernelEnd)
			kernelEnd = time;
		if (time < kernelStart)
			kernelStart = time;
	}
	if(!firstKernelStartCaught)
	{
		firstKernelStartCaught = true;
		firstKernelStart = kernelStart;
	}
	for (int i = 0; i < threads*registers; i++)
	{
		host_tst[i].time = host_tst[i].time/clk - firstKernelStart;
	}
	kernelStart -= firstKernelStart;
	kernelEnd -= firstKernelStart;
	outfile << -1 << "," << -1 << "," << -1 << "," << kernelStart << ",start_" << kernelName << "_" << savedResultNumber << "\n";
	for (int i = 0; i < threads*registers; i++)
	{
		timestamp tstmp = host_tst[i];
		outfile << tstmp.x << "," << tstmp.y << "," << tstmp.z << "," << tstmp.time << "," << kernelName << "_" << labels[tstmp.label] << "_" << savedResultNumber <<"\n";
		//outfile << tstmp.tid << "," << tstmp.time / prop->clockRate << "," << tstmp.label << "\n";
	}
	outfile << -1 << "," << -1 << "," << -1 << "," << kernelEnd << ",end_" << kernelName << "_" << savedResultNumber << "\n";
	savedResultNumber++;
	clearBase<<<1,1>>>();
	checkCudaErrors(cudaFree(myTst));
#endif
}
