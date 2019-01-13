program:
	nvcc -w -std=c++11 shared.cu -o shared
profiler:
	nvcc -w -std=c++11 shared.cu -o shared_profiler -DENABLE_PROFILER
clean:
	rm -f *.o shared
	rm -f *.o shared_profiler
