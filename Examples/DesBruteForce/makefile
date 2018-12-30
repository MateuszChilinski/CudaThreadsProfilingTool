kernel:
	nvcc -w -std=c++11 kernel.cu -o kernel
profiler:
	nvcc -w -std=c++11 kernel.cu -o kernel_profiler -DENABLE_PROFILER
clean:
	rm -f *.o kernel
	rm -f *.o kernel_profiler
