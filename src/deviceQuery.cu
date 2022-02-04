#include <cstdio>
#include <cuda_runtime.h>

int main()
{
	cudaDeviceProp prop;        // device properties
	int count;                  // device count
	cudaGetDeviceCount(&count); // get device count
	
	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i); // get device properties

		printf(" --- General Information for device %d ---\n", i);
		printf("Name                     : %s\n", prop.name);
		printf("Compute capability       : %d.%d\n", prop.major, prop.minor);
		printf("Clock rate               : %.4f MHz\n", prop.clockRate / 1000000.0f);
		printf("Device copy overlap      : %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
		printf("Kernel execition timeout : %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

		printf("\n --- Memory Information for device %d ---\n", i);
		printf("Total global memory   : %.4f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
		printf("Total constant memory : %.4f KB\n", prop.totalConstMem / 1024.0f);
		printf("Max memory pitch      : %.4f GB\n", prop.memPitch / (1024.0f * 1024.0f * 1024.0f));
		printf("Texture Alignment     : %llu\n", prop.textureAlignment);

		printf("\n --- MP Information for device %d ---\n", i);
	 	printf("Multiprocessor count  : %d\n", prop.multiProcessorCount);
	 	printf("Shared memory per MP  : %.4f KB\n", prop.sharedMemPerBlock / 1024.0f);
	 	printf("Registers per MP      : %d\n", prop.regsPerBlock);
	 	printf("Threads in warp       : %d\n", prop.warpSize);
	 	printf("Max threads per block : %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions   : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("----------------------------------------\n");
	}
}