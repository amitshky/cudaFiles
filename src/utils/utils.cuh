#pragma once

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef DEBUG_MODE // defined in CMakeLists.txt
	#define cudaCheckError(val) CheckCuda(val, #val, __FILE__, __LINE__)
#else
	#define cudaCheckError(val) val
#endif

inline void CheckCuda(cudaError_t error, const char* func, const char* file, const int line)
{
	if (error)
	{
		printf("CUDA::ERROR::%d in function: `%s` file: `%s` on line: %d",
			error, func, file, line);
		std::exit(-1);
	}
}