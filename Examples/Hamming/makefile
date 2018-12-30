program:
	nvcc -w -std=c++11 hamming.cu -o hamming
profiler:
	nvcc -w -std=c++11 hamming.cu -o hamming_profiler -DENABLE_PROFILER
clean:
	rm -f *.o hamming
	rm -f *.o hamming_profiler
