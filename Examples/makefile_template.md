# Template

```
program:
	nvcc -w -std=c++11 <input> -o <outputfile>
profiler:
	nvcc -w -std=c++11 <input> -o <outputfile> -DENABLE_PROFILER
clean:
	//cleanup
```

# Example
```
program:
	nvcc -w -std=c++11 kernel.cu -o kernel
profiler:
	nvcc -w -std=c++11 kernel.cu -o kernel_profiler -DENABLE_PROFILER
clean:
	rm -f *.o kernel
	rm -f *.o kernel_profiler
```

# USAGE
`make program` - compiles program with profiler disabled

`make profiler` - compiles program with profiler enabled
