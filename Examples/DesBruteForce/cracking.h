#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "../../Library/ParallelThreadProfiler.cuh"

__device__ __host__
uint64_t generate_key(uint64_t number)
{
	uint64_t key = 0;

	for (int i = 0; i < 8; i++)
	{
		key = key | ((number >> (i * 7)) & 0b01111111) << (i * 8 + 1);
	}

	return key;
}

__host__
uint64_t generate_message(char start_char, char data[], int alphabet_size)
{
	uint64_t message = 0;

	for (int i = 0; i < 8; i++)
	{
		if (data[i] - start_char > alphabet_size - 1)
		{
			return 0;
		}
		message = message | (uint64_t)data[i] << (56 - i * 8);
	}

	return message;
}

__device__
uint64_t generate_message_kernel(char start_char, int alphabet_size, uint64_t iteration)
{
	uint64_t message = 0;

	for (int i = 0; i < 8; i++)
	{
		message = message | (uint64_t)(start_char + iteration % alphabet_size) << (56 - i * 8);
		iteration /= alphabet_size;
	}

	return message;
}

__global__
void des_kernel(uint64_t cypher, int key_length, char start_char, int alphabet_size, uint64_t* value, bool* found)
{
	RegisterTimeMarker(0);
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_count = blockDim.x * gridDim.x;
	uint64_t keys_count = 1 << (key_length + 1);
	// wiem że to niżej jest głupie
	uint64_t messages_count = alphabet_size * alphabet_size * alphabet_size * alphabet_size * alphabet_size * alphabet_size * alphabet_size * alphabet_size;
	uint64_t message, key, x = 0;

	if (tid == 0)
	{
		(*found) = false;
		(*value) = 0;
	}

	for (int i = 0; (tid + i * thread_count) < (messages_count * keys_count); i++)
	{
		message = generate_message_kernel(start_char, alphabet_size, (tid + i * thread_count) / keys_count);
		key = generate_key((tid + i * thread_count) % keys_count);

		x = des(message, key, true);

		if (x == cypher)
		{
			*found = true;
			*value = key;
		}
		if (*found)
		{
			RegisterTimeMarker(1);
			return;
		}
	}
	RegisterTimeMarker(1);
}

void run_kernel(int blocks, int threads, uint64_t cypher, uint64_t key, int key_length, char start_char, int alphabet_size, char data[])
{
	uint64_t value;
	bool found;
	bool* d_found;
	uint64_t* d_value;

	gpuErrchk(cudaMalloc((void**)&d_found, sizeof(bool)));
	gpuErrchk(cudaMalloc((void**)&d_value, sizeof(uint64_t)));

	cudaEvent_t start, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));

	std::cout << "Kernel starts!\n";
	gpuErrchk(cudaEventRecord(start, 0));
	ParallelThreadProfiler::CreateLabel("start",0);
	ParallelThreadProfiler::CreateLabel("end",1);
	ParallelThreadProfiler::InitialiseKernelProfiling("des",(blocks*threads),2);
	des_kernel <<<blocks, threads>> > (cypher, key_length, start_char, alphabet_size, d_value, d_found);
	ParallelThreadProfiler::SaveResults();	
	gpuErrchk(cudaPeekAtLastError());	// ivalid launch argument check
	gpuErrchk(cudaDeviceSynchronize());	// kernel execution error

	gpuErrchk(cudaEventRecord(stop, 0));
	gpuErrchk(cudaEventSynchronize(stop));

	float time;
	gpuErrchk(cudaEventElapsedTime(&time, start, stop));
	std::cout << "Kernel time: " << time << "ms\n";

	// symbole totalnie nie dzialaja tak jak powinny
	//cudaMemcpyFromSymbol(&found, &d_found, sizeof(d_found), 0, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));

	if (found)
	{
		//cudaMemcpyFromSymbol(&value, &d_value, sizeof(d_value), 0, cudaMemcpyDeviceToHost);
		gpuErrchk(cudaMemcpy(&value, d_value, sizeof(uint64_t), cudaMemcpyDeviceToHost));
		std::cout << "Key was found: ";
		bin_print(value);
		std::cout << "Key is: " << (value == key ? "correct!\n" : "incorrect!!!\n");
	}
	else
	{
		std::cout << "Key wasn't found!\n";
	}

	gpuErrchk(cudaFree(d_found));
	gpuErrchk(cudaFree(d_value));
}