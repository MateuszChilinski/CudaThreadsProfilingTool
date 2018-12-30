#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <random>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <bitset>

#include "../../Library/CudaThreadProfiler.cuh"

using namespace std;



void CheckErrors(cudaError_t status);
bool CompareResults(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result);

template<unsigned long long k>
class SequenceOfBits;

//Number of bits per sequence
const unsigned long long sequence_length = 10000;

//Number of sequences
const unsigned long long input_sequence_count = 100000ull;
const unsigned long long comparisons = (((input_sequence_count*(input_sequence_count - 1)) / 2));

SequenceOfBits<sequence_length> * GenerateInput();
SequenceOfBits<sequence_length> * GenerateInputConstant();
vector<pair<int, int> > ToPairVector(const SequenceOfBits<comparisons> & result_sequence);
void HammingCPU(SequenceOfBits<sequence_length> * sequence, SequenceOfBits<comparisons> * out_data);
vector<pair<int, int> > FindPairsGPU(SequenceOfBits<sequence_length> * h_sequence);
vector<pair<int, int> > FindPairsCPU(SequenceOfBits<sequence_length> * sequence);
void PrintSequences(SequenceOfBits<sequence_length> * sequences);




template<unsigned int N>
class DeviceResults;
class Results
{
public:
	unsigned int **result_array;
};

//Host holds the array
template<unsigned int N>
class HostResults : public Results
{
public:

	HostResults()
	{
		//N - 1 is equal to the number of rows. No need to create row with 0 fields
		result_array = new unsigned int* [N - 1];

		for (int i = 0; i < N - 1; i++)
		{
			//Results are from ballot, which returns 32bits. That is why we are counting how many full 32bit words we need for each row 
			int number_of_words = (int)(ceil((i + 1) / 32.0));			
			result_array[i] = new unsigned int[number_of_words];
		}
	}

	

	char GetBit(unsigned int row, unsigned int col) const
	{
		//We never use row 0 , as it is left down triangle matrix without main diagonal, so row 0 is empty. No need to create it then
		//We hold each column in one bit, so we need to get it by & with 1 moved to the correct spot
		return (char)(result_array[row - 1][col / 32] >> (col % 32) & 1);
	}

	HostResults(HostResults<N> &&h_result)
	{
		this->result_array = h_result.result_array;
		h_result.result_array = nullptr;
	}

	HostResults<N>&& operator=(HostResults<N> &&h_result)
	{
		this->result_array = h_result.result_array;
		h_result.result_array = nullptr;
	}

	~HostResults()
	{
		if (result_array == nullptr)
			return;

		for (int i = 0; i < N - 1; i++)
		{
			delete[] result_array[i];
		}

		delete[] result_array;
	}
	
};

//similar like HostResults, but for device . Device copies their result to it after they finish checking
template<unsigned int N>
class DeviceResults : public Results
{
public:

	HostResults<N> ToHostArray()
	{
		HostResults<N> host;
		unsigned int * temporal[N - 1];
		CheckErrors(cudaMemcpy(temporal, result_array, sizeof(unsigned int*)*(N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; ++i)
		{
			unsigned int number_of_words = (unsigned int)(ceil((i + 1) / 32.0));
			CheckErrors(cudaMemcpyAsync(host.result_array[i], temporal[i], sizeof(unsigned int) * number_of_words , cudaMemcpyDeviceToHost));
		}
		CheckErrors(cudaDeviceSynchronize());
		return host;
	}

	DeviceResults()
	{
		//N - 1 is equal to the number of rows. No need to create row with 0 fields
		CheckErrors(cudaMalloc(&result_array, sizeof(unsigned int* ) * (N - 1)));

		unsigned int* temporal[N - 1];
		for (int i = 0; i < N - 1; ++i)
		{
			double length = (ceil((i + 1) / 32.0));
			CheckErrors(cudaMalloc(&(temporal[i]), sizeof(unsigned int) * length));
			CheckErrors(cudaMemset(temporal[i], 0, sizeof(unsigned int) * length));
		}
		CheckErrors(cudaMemcpyAsync(result_array, &(temporal[0]), sizeof(unsigned int*) * (N - 1), cudaMemcpyHostToDevice));
		CheckErrors(cudaDeviceSynchronize());
	}

	

	~DeviceResults()
	{
		unsigned int *temporal[N - 1];
		CheckErrors(cudaMemcpy(temporal, result_array, sizeof(unsigned int*) * (N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; i++)
		{
			CheckErrors(cudaFree(temporal[i]));
		}
		CheckErrors(cudaFree(result_array));
	}
};

template<unsigned int N>
vector<pair<int, int> > GetResultPairs(const HostResults<N> & result_array);

__host__ __device__ char compareSequences(SequenceOfBits<sequence_length> * sequence1, SequenceOfBits<sequence_length> * sequence2);
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j);
__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j);

const unsigned long long threads_in_block = 1024;
const unsigned long long rows_per_call = 35;
const unsigned long long words64bits_in_sequence = ((sequence_length + 63) / 64);

int main()
{
	cudaError_t cudaStatus;
	//Need more shared memory than the default allocation
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	printf("Generation sequence in progress...");
	cout.flush();
	SequenceOfBits<sequence_length>* sequence = GenerateInputConstant();
	printf("Completed!\n");

	CudaThreadProfiler::InitialiseProfiling();
	
	printf("Started searching for pairs of sequences with Hamming distance equal 1 on GPU...\n");
	vector<pair<int, int>> resultsGPU = FindPairsGPU(sequence);
	printf("Completed!\n");

	printf("Started searching for pairs of sequences with Hamming distance equal 1 on CPU...\n");
	vector<pair<int, int>> resultsCPU = FindPairsCPU(sequence);
	printf("Completed!\n");

	printf("Comparing GPU results with CPU results...\n");
	CompareResults(resultsGPU, resultsCPU);
	//PrintSequences(sequence);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

template<unsigned long long N>
class SequenceOfBits
{
public:
	// !!(N%64) returns 0 if N is divisible by 64 and 1 if it is not. Array must contain whole 64-bits long words.
	static const unsigned long long array_size = (N / 64 + (!!(N % 64))) * 8;

	__host__ __device__ SequenceOfBits() {}

	__host__ __device__ SequenceOfBits(const SequenceOfBits<N> & sequence)
	{
		memcpy(array, sequence.array, array_size * 8);
	}

	__host__ __device__ const SequenceOfBits<N> & operator=(const SequenceOfBits<N> & sequence)
	{
		memcpy(array, sequence.array, array_size * 8);
		return sequence;
	}

	//Get a 32-bits long word, so 32/8 = 4 bytes long.
	__host__ __device__ inline unsigned int *Get32BitsWord(unsigned int word_index)
	{
		return (unsigned int*)(array + word_index * (32 / 8));
	}

	//Get a 64-bits long word , so 64/8 = 8 bytes long word
	__host__ __device__ inline unsigned long long *Get64BitsWord(unsigned long long word_index)
	{
		return (unsigned long long*)(array + word_index * (64 / 8));
	}

	//Char has 1 byte so 8 bits. Divide by 8 to get the byte with our searched bit,
	//move it to the right so our bit is the least significant bit of the byte.
	//Then we can & it with 1 to get the value.
	__host__ __device__ inline char GetBit(unsigned long long index) const
	{
		return array[index / 8] >> (index % 8) & 1;
	}

	//Can't write only one bit, have to write whole byte. Get the byte containing our bit , get a mask of it with ones everywhere and zero on the bit spot.
	//After &-ing the mask with the byte we get the same byte, but with 0 in place of our bit.
	//Afterwards we can & it with a byte containing zeroes everywhere and our desired value in the place of the searched bit to write the correct value in the desired bit spot.
	//!! makes 0 from 0 and 1 from non-zero value
	__host__ __device__ inline void SetBit(unsigned long long index, char value)
	{
		array[index / 8] = (array[index / 8] & (~(1 << (index % 8)))) | ((!!value) << (index % 8));
	}		
private:
	char array[array_size];
};

__host__ __device__ char compareSequences(SequenceOfBits<sequence_length> * first_sequence, SequenceOfBits<sequence_length> * second_sequence)
{
	int difference_count = 0;
	//Words are 64bits, and (sequence_length + 63) / 64 works as ceil(sequence_length/64)
	for (int i = 0; i < (sequence_length + 63) / 64; ++i)
	{
		unsigned long long int first_word, second_word, xor_result;
		first_word = *(first_sequence->Get64BitsWord(i));
		second_word = *(second_sequence->Get64BitsWord(i));

		xor_result = first_word ^ second_word;
		//if xor_result & (xor_result - 1) == 0 that means that was not a power of 2, so we stop. Otherwise that means there was a difference on exactly one place.
		difference_count += xor_result == 0 ? 0 : (xor_result & (xor_result - 1) ? 2 : 1);

		if (difference_count > 1)		
			return 0;
		
	}
	//returns 0 if difference_count = 0 and 1 otherwise
	return !!difference_count;
}

//Function to get the (x, y) coordinates from a N x N matrix based on the index. We care only about what is above the main diagonal.
//On the main diagonal we have a comparison with themselfs (H(a, a) = 0 ), and below it we have duplicates, as H(a, b) == H(b, a).
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j)
{
	//adding 1 to k to skip first result
	*i = (unsigned int)ceil((0.5 * (-1 + sqrt((double)(1 + 8 * (k + 1))))));
	//decreasing 1 from j , as we start from 0 not 1
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((unsigned long long)(*i) - 1)) - 1;
}
//Function to get the index from (x, y) coordinates
__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j)
{
	return ((unsigned long long)i) * (i - 1) / 2 + j;
}

void HammingCPU(SequenceOfBits<sequence_length> * sequence, SequenceOfBits<comparisons> * out_data)
{
	int x = 1, y = 0;
	for (unsigned long long k = 0; k < comparisons / 32; ++k)
	{
		unsigned int result = 0;
		for (int i = 0; i < 32; ++i)
		{
			//Setting the result one bit at the time. compareSequences returns 0 or 1, so we set the value in the correct place in the result.
			result |= (unsigned int)(compareSequences(sequence + x, sequence + y)) << i;
			++y;
			if (y == x)
			{
				y = 0;
				++x;			
			}
		}
		*(out_data->Get32BitsWord(k)) = result;
	}
	//Something left, not a whole word
	if (comparisons % 32)
	{
		unsigned int result = 0;
		//on the missing places there will be zeroes
		for (int i = 0; i < comparisons % 32; i++)
		{
			result |= (unsigned int)(compareSequences(sequence + x, sequence + y)) << i;
			++y;
			if (y == x)
			{
				y = 0;
				++x;				
			}
		}
		
		*(out_data->Get32BitsWord(comparisons / 32)) = result;
	}
}


bool CompareResults(const vector<pair<int, int>>& gpu_result, const vector<pair<int, int> > & cpu_result)
{
	unsigned long long shorter_vector_length;
	unsigned long long cpu_pair_count = cpu_result.size();
	unsigned long long gpu_pair_count = gpu_result.size();

	if (gpu_pair_count < cpu_pair_count)
		shorter_vector_length = gpu_pair_count;
	else
		shorter_vector_length = cpu_pair_count;

	vector<pair<int, int>> result_gpu(gpu_result);
	vector<pair<int, int>> result_cpu(cpu_result);

	//sorting to make sure the pairs are in the same order, to be able to compare the.
	sort(result_cpu.begin(), result_cpu.end());
	sort(result_gpu.begin(), result_gpu.end());
	

	const vector<pair<int, int> > & longer_vector = cpu_pair_count > gpu_pair_count ? result_cpu : result_gpu;
	bool equal = true;
	
	if (gpu_pair_count != cpu_pair_count)
	{
		cout << "Number of elements in both results is not equal (GPU: " << gpu_pair_count << ", CPU: " << cpu_pair_count << ") !" << endl;
		equal = false;
	}
	else
	{
		cout << "Number of elements in both result is equal (GPU: " << gpu_pair_count << ", CPU: " << cpu_pair_count << ")" << endl;
	}

	//need to have access to the last number that was checked
	int i = 0;
	for (; i < shorter_vector_length; ++i)
	{
		if (result_gpu[i] != result_cpu[i])
		{
			equal = false;
			//cout << "Difference on pair number " << i << "; GPU Pair: (" << result_gpu[i].first << ", " << result_gpu[i].second << ") CPU Pair: ("
			//	<< result_cpu[i].first << ", " << result_cpu[i].second << ")" << endl;
		}		

	}
	if (cpu_pair_count != gpu_pair_count)
	{
		cout << "Remaining pairs from " << ((cpu_pair_count > gpu_pair_count) ? "CPU" : "GPU") << " result:" << endl;
		for (; i < longer_vector.size(); ++i)
		{
			//cout << "(" << longer_vector[i].first << ", " << longer_vector[i].second << ")" << endl;
		}
	}

	if (equal)
		printf("Results are the same!\n");

	return equal;
}

SequenceOfBits<sequence_length> * GenerateInput()
{
	//Mersene Twister 64 bits
	mt19937_64 source;
	int seed = random_device()();
	source.seed(seed);
	SequenceOfBits<sequence_length> * result = new SequenceOfBits<sequence_length>[input_sequence_count];

	memset(result, 0, sizeof(SequenceOfBits<sequence_length>) * input_sequence_count);
	

	for (int i = 0; i < input_sequence_count; ++i)
	{
		//generate it 64 bits at the time
		for (int j = 0; j < SequenceOfBits<sequence_length>::array_size / 8 - 1; ++j)
		{
			*(result[i].Get64BitsWord(j)) = source();
		}
		//last word can be not full, so we generate in separately. We move the 64bit word so we are left with only the required number of set bits and with the rest of them set to 0
		*(result[i].Get64BitsWord(sequence_length / 64)) = source() >> (64 - (sequence_length % 64));
	}	

	return result;
}

SequenceOfBits<sequence_length> * GenerateInputConstant()
{

	SequenceOfBits<sequence_length> * result = new SequenceOfBits<sequence_length>[input_sequence_count];

	memset(result, 0, sizeof(SequenceOfBits<sequence_length>) * input_sequence_count);

	 for(int i=0; i < input_sequence_count; ++i)
	{
		*(result[i].Get32BitsWord(0)) = i;
	}
	return result;
}
vector<pair<int, int> > ToPairVector(const SequenceOfBits<comparisons> & result_sequence)
{
	vector<pair<int, int> > resultVector;
	int x = 1, y = 0;
	for (unsigned long long k = 0; k < comparisons; k++)
	{
		if (result_sequence.GetBit(k))
		{
			resultVector.push_back(make_pair(x, y));
		}

		++y;
		if (y == x)
		{
			y = 0;
			++x;
		}

	}

	return resultVector;
}

vector<pair<int, int> > FindPairsCPU(SequenceOfBits<sequence_length> * sequence)
{
	SequenceOfBits<comparisons> *results_array;
	results_array = new SequenceOfBits<comparisons>();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	HammingCPU(sequence, results_array);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float result_time;
	cudaEventElapsedTime(&result_time, start, stop); 
	printf("CPU execution time: %f\n", result_time);
	vector<pair<int, int>> result = ToPairVector(*results_array);
	delete results_array;
	return result;
}






__global__ void HammingGPU(SequenceOfBits<sequence_length> *sequences, unsigned int **array, unsigned int row_offset, unsigned int column_offset)
{
	RegisterTimeMarker("start");
	unsigned int sequence_number = threadIdx.x + blockIdx.x * blockDim.x + column_offset;

	char result[rows_per_call];
	memset(result, 0, rows_per_call * sizeof(char));

	//Get the correct number of rows. Offset is from 0, so if we have less than rows_per_call rows, we take the rest
	char number_of_rows;
	
	if (rows_per_call > row_offset)
		number_of_rows = row_offset;
	else
		number_of_rows = rows_per_call;


	SequenceOfBits<sequence_length> & sequence1 = *(sequences + sequence_number);

	__shared__ SequenceOfBits<sequence_length> shared_array[rows_per_call];

	for (unsigned int offset = 0; offset < number_of_rows * words64bits_in_sequence; offset += blockDim.x)
	{
		unsigned int new_id = threadIdx.x + offset;
		//Copy
		if (new_id < number_of_rows * words64bits_in_sequence)
			*(shared_array[new_id / words64bits_in_sequence].Get64BitsWord(new_id % words64bits_in_sequence)) =	*((sequences + row_offset - new_id / words64bits_in_sequence)->Get64BitsWord(new_id % words64bits_in_sequence));		
	}
	__syncthreads();
	//Comparing not whole, but 64bits at the time.
	for (int j = 0; j < words64bits_in_sequence; ++j)
	{
		unsigned long long sequence1_part = *(sequence1.Get64BitsWord(j));
		for (int i = 0; i < number_of_rows; ++i)
		{
			//if the difference > 1, there is no point in calculating it further,
			//the final distance will be at least equal to current difference
			if (result[i] <= 1)
			{
				unsigned long long part = *(shared_array[i].Get64BitsWord(j));
				unsigned long long xor_result = sequence1_part ^ part;
				result[i] += (xor_result ? ((xor_result & (xor_result - 1)) ? 2 : 1) : 0);
			}
		}
	}

	for (int i = 0; i < number_of_rows; ++i)
	{
		unsigned int sequence2_number = row_offset - i;
		__syncthreads();
		unsigned int voting_result = __ballot(result[i] == 1);

		if (sequence2_number > sequence_number && !(sequence_number % 32))
		{
			* (array[sequence2_number - 1] + sequence_number / 32) = voting_result;
		}
	}

	RegisterTimeMarker("end");
}

vector<pair<int, int> > FindPairsGPU(SequenceOfBits<sequence_length> * h_sequence)
{
	SequenceOfBits<sequence_length> *d_idata;
	DeviceResults<input_sequence_count> d_result;
	cudaEvent_t start_memory, stop_memory, start_execution, stop_execution;
	cudaEventCreate(&start_memory);	
	cudaEventCreate(&start_execution);
	cudaEventCreate(&stop_memory);
	cudaEventCreate(&stop_execution);
	float execution_time, memory_time;

	unsigned long long inputSize = sizeof(SequenceOfBits<sequence_length>)* input_sequence_count;

	cudaEventRecord(start_memory);
	cudaEventSynchronize(start_memory);

	CheckErrors(cudaMalloc(&d_idata, inputSize));
	CheckErrors(cudaMemcpy(d_idata, h_sequence, inputSize, cudaMemcpyHostToDevice));
	cudaEventRecord(start_execution);
	cudaEventSynchronize(start_execution);
	for (int j = input_sequence_count - 1; j > 0; j -= rows_per_call)
	{
		if (j >= threads_in_block)
		{
			int blocks = j/threads_in_block;
			CudaThreadProfiler::InitialiseKernelProfiling((blocks*threads_in_block)/32,2);
			//start as many full blocks as you can
			HammingGPU << < blocks, threads_in_block >> > (d_idata, d_result.result_array, j, 0);
		}

		CudaThreadProfiler::SaveResults();			


		// Start the last block, which won't be full
		if (j % threads_in_block)
		{
			CudaThreadProfiler::InitialiseKernelProfiling((j%threads_in_block)/32,2);
			HammingGPU << < 1, j % threads_in_block >> > (d_idata, d_result.result_array, j, j - (j%threads_in_block));
		}

		CudaThreadProfiler::SaveResults();			

	}

	CheckErrors(cudaDeviceSynchronize());
	cudaEventRecord(stop_execution);
	cudaEventSynchronize(stop_execution);
	cudaEventElapsedTime(&execution_time, start_execution, stop_execution);

	HostResults<input_sequence_count> h_result(d_result.ToHostArray());

	cudaEventRecord(stop_memory);
	cudaEventSynchronize(stop_memory);
	cudaEventElapsedTime(&memory_time, start_memory, stop_memory);

	cudaFree(d_idata);
	printf("GPU Times : Only execution: %f, including copying to memory : %f\n", execution_time, memory_time);
	return GetResultPairs(h_result);
}

template<unsigned int N>
vector<pair<int, int> > GetResultPairs(const HostResults<N> & result_array)
{
	vector<pair<int, int> > result;
	for (int i = 1; i < N; ++i)
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



void PrintSequences(SequenceOfBits<sequence_length> * sequences)
{
	for (int i = 0; i < input_sequence_count; ++i)
	{
		for (unsigned long long j = 0; j < sequence_length; ++j)
		{
			cout << (short int)sequences[i].GetBit(j);
		}
		printf("\n");
	}
		
}

void CheckErrors(cudaError_t status)
{
	if (status != cudaSuccess)
		printf("Cuda Error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(status));
}
