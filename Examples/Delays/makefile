program:
	nvcc -w -std=c++11 delays.cu -o delays
profiler:
	nvcc -w -std=c++11 delays.cu -o delays_profiler -DENABLE_PROFILER
clean:
	rm -f *.o delays
	rm -f *.o delays_profiler