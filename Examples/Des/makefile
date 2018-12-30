program:
	nvcc -w -std=c++11 main.cu -o des
profiler:
	nvcc -w -std=c++11 main.cu -o des_profiler -DENABLE_PROFILER
clean:
	rm -f *.o des
	rm -f *.o des_profiler
