program:
	nvcc -w -std=c++11 single_loop.cu -o single_loop
profiler:
	nvcc -w -std=c++11 single_loop.cu -o single_loop_profiler -DENABLE_PROFILER
clean:
	rm -f *.o single_loop
	rm -f *.o single_loop_profiler
