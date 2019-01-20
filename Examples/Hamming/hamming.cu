#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <ctime>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "../../Library/ParallelThreadProfiler.cuh"

using namespace std;

const int K = 1000; // No bits in seq
const unsigned long long N = 10000ull; // No seq
const unsigned long long cmp = (N * (N - 1)) / 2; // No comparisons
const int threadsPerBlock = 1024;
const int seqPerCall = 35; // no rows per func call
const int bit64words = ((K + 63) / 64);

template<unsigned long long k>
class Sequence;

class PairVector {
public:
	PairVector() {

	}

	PairVector&& operator=(PairVector &&pVec)
	{
		this->vec = pVec.vec;
	}

	PairVector(const PairVector &pVec)
	{
		this->vec = pVec.vec;
	}

	static bool ComparePairs(const PairVector &pVec1, const PairVector &pVec2)
	{
		unsigned long long size1 = pVec1.vec.size(), size2 = pVec2.vec.size();
		unsigned long long n = size1 < size2 ? size1 : size2;

		vector<pair<int, int> > vec1(pVec1.vec);
		vector<pair<int, int> > vec2(pVec2.vec);
		sort(vec1.begin(), vec1.end());
		sort(vec2.begin(), vec2.end());

		bool equal = true;
		if (size1 != size2)
		{
			cout << "Number of elements differs (" << size1 << "=/=" << size2 << ") !" << endl;
			equal = false;
		}
		else
		{
			cout << "Number of elements are equal (" << size1 << "==" << size2 << ")" << endl;
		}

		int i;
		for (i = 0; i < n; ++i)
		{
			if (vec1[i] != vec2[i])
			{
				cout << "There is a difference on index " << i << ": (" << vec1[i].first << ", " << vec1[i].second << ") =/= ("
					<< vec2[i].first << ", " << vec2[i].second << ")" << endl;
				equal = false;
			}
		}

		if (equal)
			cout << "Results are the same" << endl;
		else
			cout << "Results aren't the same" << endl;

		return equal;
	}

	void push_back(const pair<int, int>& p)
	{
		vec.push_back(p);
	}

	unsigned long long size()
	{
		return vec.size();
	}

private:
	vector<pair<int, int> > vec;
};

void CUDAErrorChecker(cudaError_t st)
{
	if (cudaSuccess != st)
		cout << "Error in CUDA " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(st) << endl;
}

template<unsigned int n>
class HostResultArray;

template<unsigned int n>
class DeviceResultArray
{
public:
	unsigned int **deviceArray;

	DeviceResultArray()
	{
		CUDAErrorChecker(cudaMalloc(&deviceArray, sizeof(unsigned int*)*(n - 1)));
		unsigned int* tArray[n - 1];
		for (int i = 0; i < n - 1; ++i)
		{
			CUDAErrorChecker(cudaMalloc(&(tArray[i]), sizeof(unsigned int) * (ceil((i + 1) / 32.0))));
			CUDAErrorChecker(cudaMemset(tArray[i], 0, sizeof(unsigned int) * (ceil((i + 1) / 32.0))));
		}
		CUDAErrorChecker(cudaMemcpy(deviceArray, tArray, sizeof(unsigned int*)*(n - 1), cudaMemcpyHostToDevice));
		CUDAErrorChecker(cudaDeviceSynchronize());
	}

	~DeviceResultArray()
	{
		unsigned int *tArray[n - 1];
		CUDAErrorChecker(cudaMemcpy(tArray, deviceArray, sizeof(unsigned int*)*(n - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < n - 1; i++)
		{
			CUDAErrorChecker(cudaFree(tArray[i]));
		}
		CUDAErrorChecker(cudaFree(deviceArray));
	}

	HostResultArray<n> ToHostArray()
	{
		HostResultArray<n> host;
		unsigned int * tArray[n - 1];
		CUDAErrorChecker(cudaMemcpy(tArray, deviceArray, sizeof(unsigned int*)*(n - 1), cudaMemcpyDeviceToHost));

		for (int i = 0; i < n - 1; ++i)
		{
			CUDAErrorChecker(cudaMemcpy(host.hostArray[i], tArray[i], sizeof(unsigned int) * (unsigned int)(ceil((i + 1) / 32.0)), cudaMemcpyDeviceToHost));
		}
		CUDAErrorChecker(cudaDeviceSynchronize());

		return host;
	}
};

template<unsigned int n>
class HostResultArray
{
public:
	unsigned int **hostArray;

	HostResultArray()
	{
		hostArray = new unsigned int*[n - 1];

		for (int i = 0; i < n - 1; i++)
		{
			hostArray[i] = new unsigned int[(int)(ceil((i + 1) / 32.0))];
		}
	}

	~HostResultArray()
	{
		if (hostArray == nullptr)
			return;

		for (int i = 0; i < n - 1; i++)
		{
			delete[] hostArray[i];
		}

		delete[] hostArray;
	}

	HostResultArray<n>&& operator=(HostResultArray<n> &&h_result)
	{
		this->hostArray = h_result.hostArray;
		h_result.hostArray = nullptr;
	}

	HostResultArray(HostResultArray<n> &&h_result)
	{
		this->hostArray = h_result.hostArray;
		h_result.hostArray = nullptr;
	}

	void CopyRows(const DeviceResultArray<n> & array, unsigned int start, unsigned int quantity)
	{
		unsigned int **tArray = new unsigned int*[quantity];
		cudaMemcpy(tArray, array.deviceArray + start - 1, quantity * sizeof(unsigned int*), cudaMemcpyDeviceToHost);

		for (int i = 0; i < quantity; ++i)
		{
			cudaMemcpy(hostArray[start - 1 + i], tArray[i], sizeof(unsigned int) * (int)(ceil((start + i) / 32.0)), cudaMemcpyDeviceToHost);
		}

		delete[] tArray;
	}

	char GetBit(unsigned int row, unsigned int col) const
	{
		return (char)(hostArray[row - 1][col / 32] >> (col % 32) & 1);
	}
};

void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j);
char compareSequences(Sequence<K> * sequence1, Sequence<K> * sequence2);
void CPUHamming(Sequence<K> * sequence, Sequence<cmp> * odata);
PairVector CPUFindPairs(Sequence<K> * sequence);

template<unsigned int N>
PairVector ToPairVectorH(const HostResultArray<N> & result_array);
PairVector ToPairVector(const Sequence<cmp> & result_sequence);

Sequence<K> * GenerateInput();

PairVector GPUFindPairs(Sequence<K> * h_sequence);

int main()
{
	cudaError_t cudaStatus;
	cout << "Sequence generation started..." << endl;
	Sequence<K>* sequence = GenerateInput();
	cout << "Sequence generation ended!" << endl;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	cout << endl << "GPU computations started..." << endl;
	PairVector gpuRes = GPUFindPairs(sequence);
	cout << "GPU computations ended!" << endl << "GPU has found " << gpuRes.size() << " results" << endl;
	cout.flush();

	cout << endl << "CPU computation started..." << endl;
	PairVector cpuRes = CPUFindPairs(sequence);
	cout << "CPU computations ended!" << endl << "CPU has found " << cpuRes.size() << " results" << endl;
	cout.flush();

	PairVector::ComparePairs(gpuRes, cpuRes);

	delete[] sequence;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

template<unsigned long long k>
class Sequence
{
public:
	__host__ __device__ Sequence()
	{

	}
	__host__ __device__ inline void SetBit(unsigned long long index, char value)
	{
		arr[index / 64] = (arr[index / 64] & (~(1ull << (index % 64)))) | (((bool) value) << (index % 64));
	}
	__host__ __device__ inline char GetBit(unsigned long long index) const
	{
		return (arr[index / 64] >> (index % 64)) & 1;
	}

	__host__ __device__ inline unsigned long long *Get64BitWord(unsigned int word_index)
	{
		return arr + word_index;
	}
	__host__ __device__ inline unsigned int *Get32BitWord(unsigned int word_index)
	{
		return ((unsigned int*)arr) + word_index;
	}

	__host__ __device__ Sequence(const Sequence<k> & sequence)
	{
		memcpy(arr, sequence.arr, size * 8);
	}

	__host__ __device__ const Sequence<k> & operator=(const Sequence<k> & sequence)
	{
		memcpy(arr, sequence.arr, size * 8);
		return sequence;
	}
	static const unsigned long long size = (k + 63) / 64;
private:
	unsigned long long arr[size];
};

/* //////////////// CPU ///////////////// */

void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j)
{
	*i = (unsigned int)ceil((0.5 * (-1 + sqrt(1 + 8 * (k + 1)))));
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((unsigned long long)(*i) - 1)) - 1;
}
char compareSequences(Sequence<K> * seq1, Sequence<K> * seq2)
{
	int difference = 0;
	for (int j = 0; j < (K + 63) / 64; ++j)
	{
		int result;
		unsigned long long int val1, val2, xorResult;
		val1 = *(seq1->Get64BitWord(j));
		val2 = *(seq2->Get64BitWord(j));

		xorResult = val1 ^ val2;
		result = xorResult != 0 ? (xorResult & (xorResult - 1) ? 2 : 1) : 0;

		difference += result;

		if (difference > 1)
			return 0;
	}

	return (bool)difference;
}

void CPUHamming(Sequence<K> * seq, Sequence<cmp> * outData)
{
	unsigned long long noComparisons = cmp;
	int k1 = 1, k2 = 0;
	for (unsigned long long i = 0; i < noComparisons / 32; ++i)
	{
		unsigned int result = 0;
		for (int j = 0; j < 32; j++)
		{
			result |= (unsigned int)(compareSequences(seq + k1, seq + k2)) << j;
			k2++;

			if (k2 == k1)
			{
				k1++;
				k2 = 0;
			}
		}
		*(outData->Get32BitWord(i)) = result;
	}

	if (noComparisons % 32)
	{
		unsigned int result = 0;
		for (int i = 0; i < noComparisons % 32; i++)
		{
			result |= (unsigned int)(compareSequences(seq + k1, seq + k2)) << i;
			k2++;

			if (k2 == k1)
			{
				k1++;
				k2 = 0;
			}
		}
		*(outData->Get32BitWord(noComparisons / 32)) = result;
	}
}

PairVector CPUFindPairs(Sequence<K> * seq)
{
	Sequence<cmp> *outData;
	outData = new Sequence<cmp>();

	float ms;
	cudaEvent_t start, stop;
	CUDAErrorChecker(cudaEventCreate(&start));
	CUDAErrorChecker(cudaEventCreate(&stop));
	CUDAErrorChecker(cudaEventRecord(start));
	CUDAErrorChecker(cudaEventSynchronize(start));

	CPUHamming(seq, outData);

	CUDAErrorChecker(cudaEventRecord(stop));
	CUDAErrorChecker(cudaEventSynchronize(stop));
	CUDAErrorChecker(cudaEventElapsedTime(&ms, start, stop));

	printf("CPU time: %f\n", ms);

	PairVector result = ToPairVector(*outData);
	delete outData;

	return result;
}

template<unsigned int n>
PairVector ToPairVectorH(const HostResultArray<n> & result_array)
{
	PairVector result;
	for (int i = 1; i < n; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (result_array.GetBit(i, j))
			{
				result.push_back(make_pair(i, j));
			}
		}
	}

	return result;
}
PairVector ToPairVector(const Sequence<cmp> & result_sequence)
{
	PairVector result;
	for (unsigned long long k = 0; k < cmp; k++)
	{
		if (result_sequence.GetBit(k))
		{
			unsigned int i, j;
			k2ij(k, &i, &j);
			result.push_back(make_pair(i, j));
		}
	}

	return result;
}


Sequence<K> * GenerateInput()
{
	//srand(time(NULL));
	srand(2018);

	Sequence<K> * r = new Sequence<K>[N];
	memset(r, 0, sizeof(Sequence<K>)*N);

	for (int i = 0; i < N; i++)
	{
		/*for (int j = 0; j < K / 32; j++)
		{
			*(r[i].Get32BitWord(j)) = rand();
		}*/
		*(r[i].Get32BitWord(0)) = 0xffffff - i;
	}

	return r;
}


/* //////////////// GPU ///////////////// */

__host__ __device__ unsigned int* GetPointer(unsigned int **arr, unsigned int row, unsigned int col)
{
	return arr[row - 1] + col / 32;
}

__global__ void GPUHamming(Sequence<K> *seq, unsigned int **arr, unsigned int row_offset, unsigned int column_offset)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int seqNumber = tid + column_offset;

	char results[seqPerCall];
	memset(results, 0, seqPerCall * sizeof(char));

	int realSeqNr = seqPerCall > row_offset ? row_offset : seqPerCall;

	Sequence<K> & s = *(seq + seqNumber);
	__shared__ Sequence<K> sharedArray[seqPerCall];

	RegisterTimeMarker(0);

	for (unsigned int offset = 0; offset < realSeqNr * bit64words; offset += blockDim.x)
	{
		unsigned int offsetId = threadIdx.x + offset;
		if (offsetId < realSeqNr * bit64words)
		{
			*(sharedArray[offsetId / bit64words].Get64BitWord(offsetId % bit64words)) =
				*((seq + row_offset - offsetId / bit64words)->Get64BitWord(offsetId % bit64words));
		}
	}

	__syncthreads();
	for (int j = 0; j < bit64words; ++j)
	{
		unsigned long long first = *(s.Get64BitWord(j));
		for (int i = 0; i < realSeqNr; ++i)
		{
			if (results[i] <= 1)
			{
				unsigned long long second = *(sharedArray[i].Get64BitWord(j));
				unsigned long long xorResult = first ^ second;
				char result = xorResult != 0 ? (xorResult & (xorResult - 1) ? 2 : 1) : 0;

				results[i] += result;
			}
		}
	}

	for (int i = 0; i < realSeqNr; ++i)
	{
		unsigned int b;
		unsigned int seq2Number = row_offset - i;
		char v = results[i] == 1;

		__syncthreads();
		b = __ballot(v);

		if (seq2Number > seqNumber && !(seqNumber % 32))
		{
			*(GetPointer(arr, seq2Number, seqNumber)) = b;
		}
	}

	RegisterTimeMarker(1);
}

PairVector GPUFindPairs(Sequence<K> * h_sequence)
{
	Sequence<K> *d_idata;
	DeviceResultArray<N> d_result;
	unsigned long long inputSize = sizeof(Sequence<K>)* N;

	float ms;
	cudaEvent_t start, stop;
	CUDAErrorChecker(cudaEventCreate(&start));
	CUDAErrorChecker(cudaEventCreate(&stop));
	CUDAErrorChecker(cudaEventRecord(start));
	CUDAErrorChecker(cudaEventSynchronize(start));

	CUDAErrorChecker(cudaMalloc(&d_idata, inputSize));
	CUDAErrorChecker(cudaMemcpy(d_idata, h_sequence, inputSize, cudaMemcpyHostToDevice));

	ParallelThreadProfiler::InitialiseProfiling();
	ParallelThreadProfiler::CreateLabel("start",0);
	ParallelThreadProfiler::CreateLabel("end",1);

	int counter =0;
	for (int i = N - 1; i > 0; i -= seqPerCall)
	{

		if (i >= threadsPerBlock)
		{
			ParallelThreadProfiler::InitialiseKernelProfiling("kernel1_"+counter, (threadsPerBlock*(i / threadsPerBlock)), 2);
			GPUHamming << < i / threadsPerBlock, threadsPerBlock >> > (d_idata, d_result.deviceArray, i, 0);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaPeekAtLastError());
			ParallelThreadProfiler::SaveResults();
		}

		if (i % threadsPerBlock > 0)
		{
			ParallelThreadProfiler::InitialiseKernelProfiling("kernel2_"+counter,i % threadsPerBlock, 2);
			GPUHamming << < 1, i % threadsPerBlock >> > (d_idata, d_result.deviceArray, i, i - (i % threadsPerBlock));
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaPeekAtLastError());
			ParallelThreadProfiler::SaveResults();
		}
	}

	CUDAErrorChecker(cudaDeviceSynchronize());
	HostResultArray<N> h_result(d_result.ToHostArray());

	CUDAErrorChecker(cudaEventRecord(stop));
	CUDAErrorChecker(cudaEventSynchronize(stop));
	CUDAErrorChecker(cudaEventElapsedTime(&ms, start, stop));

	printf("GPU time: %f\n", ms);

	PairVector res = ToPairVectorH(h_result);

	CUDAErrorChecker(cudaFree(d_idata));
	return res;
}
