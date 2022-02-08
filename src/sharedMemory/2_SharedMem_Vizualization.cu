#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include "stb_image/stb_image_write.h"

#include "utils/timer.h"
#include "utils/utils.cuh"

constexpr float PI = 3.1415926535897932f;

template<int tx, int ty>
__global__ void SharedMemViz(uint8_t* imgData, const int width, const int height, const int channels)
{
	__shared__ float sharedMem[tx][ty];
	// mapping from thread/block id to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// bounds checking
	if (x >= width || y >= height)
		return;
	
	// mapping 2D indexes to 1D image buffer
	int i = (x + y * width) * channels;

	// calc value at pixel position
	const float period = 128.0f;
	sharedMem[threadIdx.x][threadIdx.y] = 255 * (std::sinf(x * 2.0f * PI / period) + 1.0f) * (std::sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;
	
	__syncthreads(); // comment this out and see the output image

	imgData[i]     = uint8_t(0.447f * sharedMem[tx - threadIdx.x - 1][ty - threadIdx.y - 1]); // red
	imgData[i + 1] = uint8_t(0.239f * sharedMem[tx - threadIdx.x - 1][ty - threadIdx.y - 1]); // green
	imgData[i + 2] = uint8_t(0.239f * sharedMem[tx - threadIdx.x - 1][ty - threadIdx.y - 1]); // blue
	imgData[i + 3] = 255;                                                                     // alpha
}

int main()
{
	// kernel properties
	constexpr int tx = 16;
	constexpr int ty = 16;
	// image properties
	const char* outPath = "img/sharedMemViz.png";
	constexpr float aspectRatio = 1.0f / 1.0f;
	constexpr int height   = 512;
	constexpr int width    = height * aspectRatio;
	constexpr int channels = 4;
	constexpr uint64_t imgSize      = width * height * channels;
	constexpr uint64_t imgSizeBytes = imgSize * sizeof(uint8_t);
	printf("Image info:\n  width = %d\n  height = %d\n  channels = %d\n", width, height, channels);

	// allocate host and device memory
	uint8_t* imgData   = new uint8_t[imgSize];
	uint8_t* d_ImgData = nullptr;
	cudaCheckError(cudaMalloc(&d_ImgData, imgSizeBytes));

	// call cuda kernel
	const dim3 threadsPerBlock(tx, ty);
	const dim3 blocksPerGrid((tx + width - 1) / tx, (ty + height - 1) / ty);
	printf("Kernel info:\n  threads = (%d, %d)\n  blocks = (%d, %d)\nGenerating image...\n", 
		threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y);

	{
		Timer t("SharedMemViz");
		SharedMemViz<tx, ty><<<blocksPerGrid, threadsPerBlock>>>(d_ImgData, width, height, channels);
	}

	cudaCheckError(cudaGetLastError());
	// move image buffer from device to host
	cudaCheckError(cudaMemcpy(imgData, d_ImgData, imgSizeBytes, cudaMemcpyDeviceToHost));
	printf("Image generation complete.\n");

	// write image
	stbi_flip_vertically_on_write(true);
	stbi_write_png(outPath, width, height, channels, imgData, width * channels);
	printf("Image exported to %s\n", outPath);

	// clean up
	delete[] imgData;
	cudaCheckError(cudaFree(d_ImgData));
	imgData = d_ImgData = nullptr;
}