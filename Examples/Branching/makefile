program:
	nvcc -w -std=c++11 branching.cu -o branching
profiler:
	nvcc -w -std=c++11 branching.cu -o branching_profiler -DENABLE_PROFILER
clean:
	rm -f *.o branching
	rm -f *.o branching_profiler
