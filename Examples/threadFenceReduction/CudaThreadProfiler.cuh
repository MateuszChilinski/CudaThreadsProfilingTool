#pragma once
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <../../Common/helper_functions.h>
#include <../../Common/helper_cuda.h>
#include <time.h>
#include <fstream>
#include <math.h>
using namespace std; 
__global__ struct timestamp {
	int tid;
	long long time;
	char label[20];
};

__constant__ timestamp* tst;

__device__ int base = 0;

class CudaThreadProfiler
{
	static ofstream outfile;
	static timestamp* myTst;
	static int warps;
public:
	static void RegisterLabel(string);
	static void InitialiseProfiling();
	static void InitialiseKernelProfiling(int);
	static void SaveResults();
};

timestamp* CudaThreadProfiler::myTst;
ofstream CudaThreadProfiler::outfile;

int CudaThreadProfiler::warps = 0;

void CudaThreadProfiler::RegisterLabel(string newLabelName)
{
	//labels.push_back(newLabelName);
}

void CudaThreadProfiler::InitialiseProfiling()
{
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", timeinfo);
	std::string str(buffer);
	outfile.open("prof"+ str +".csv", std::ios_base::app);
}
void CudaThreadProfiler::InitialiseKernelProfiling(int warps_number)
{
	warps = warps_number;
	checkCudaErrors(cudaFree(myTst));
	checkCudaErrors(cudaMalloc((void **)&myTst, warps * sizeof(timestamp)));
	checkCudaErrors(cudaMemcpyToSymbol(tst, &myTst, sizeof(myTst)));
}
void CudaThreadProfiler::SaveResults()
{
	timestamp* host_tst = new timestamp[warps];
	cudaMemcpy(host_tst, myTst, sizeof(timestamp) * warps, cudaMemcpyDeviceToHost);
	for (int i = 0; i < warps; i++)
	{
		timestamp tstmp = host_tst[i];
		if (tstmp.tid == 0)
			continue;
		outfile << tstmp.tid << "," << tstmp.time-host_tst[0].time+1000 << "," << tstmp.label <<"\n";
	}

}
__device__ char * my_strcpy(char *dest, const char *src) {
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[i++] != 0);
	return dest;
}
__device__ void RegisterTimeMarker(char* string)
{
	if (threadIdx.x == 0)
	{
		int idx = blockIdx.x;
		int mylocation = atomicAdd(&base, 1);
		timestamp tsm;
		tsm.tid = idx;
		tsm.time = clock64();
		my_strcpy(tsm.label, string);
		tst[mylocation] = tsm;
		//printf("I am thread %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d and the time is %lld\n", idx, __mysmid(), __mywarpid(), __mylaneid(), clock64());
	}
	//printf("I am thread %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}