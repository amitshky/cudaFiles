#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include "utils/utils.cuh"


__global__ void Grayscale(uint8_t* imgData, const int width, const int height, const int channels)
{
	// mapping thread id as 2D array indexes
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// check image boundary
	if (x >= width || y >= height)
		return;
	
	// calc index of image data (1 Dim) (with offset for channels)
	int i = (x + y * width) * channels;

	// grayscale calc
	uint8_t r = 0.2126f * imgData[i];
	uint8_t g = 0.7152f * imgData[i + 1];
	uint8_t b = 0.0722f * imgData[i + 2];

	// write data to image buffer
	imgData[i]     = r + g + b; // red channel
	imgData[i + 1] = r + g + b; // green channel
	imgData[i + 2] = r + g + b; // blue channel
}

int main()
{
	// kernel properties
	constexpr int tx = 32;
	constexpr int ty = 32;
	// image properties
	const char* inPath  = "img/input.png";
	const char* outPath = "img/grayscale.png";
	int width    = 0;
	int height   = 0;
	int channels = 0;

	// load image
	stbi_set_flip_vertically_on_load(true);
	uint8_t* imgData = stbi_load(inPath, &width, &height, &channels, 0); // input image data
	uint64_t imgSize = width * height * channels;
	printf("Input image info:\n\twidth = %d, height = %d, channels = %d\n", 
		width, height, channels);

	// allocate device memory
	uint8_t* d_ImgData = nullptr;
	cudaCheckError(cudaMalloc(&d_ImgData, imgSize * sizeof(uint8_t)));
	// copy image data from host memory to device memory
	cudaCheckError(cudaMemcpy(d_ImgData, imgData, imgSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// assign grid size and block size
	const dim3 threads(tx, ty); // block size
	const dim3 blocks((width + tx - 1) / tx, (height + ty - 1)/ ty); // grid size
	printf("Kernel info:\n\tthreads = (%d, %d), blocks = (%d, %d)\n", 
		threads.x, threads.y, blocks.x, blocks.y);

	// call cuda kernel
	printf("Converting image to grayscale...\n");
	Grayscale<<<blocks, threads>>>(d_ImgData, width, height, channels);
	cudaCheckError(cudaGetLastError());
	// copy image data from device memory to host memory
	cudaCheckError(cudaMemcpy(imgData, d_ImgData, imgSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	// write image
	stbi_flip_vertically_on_write(true);
	stbi_write_png(outPath, width, height, channels, imgData, width * channels);
	printf("Image exported to %s.\n", outPath);

	// free memory
	cudaCheckError(cudaFree(d_ImgData));
	stbi_image_free(imgData);
	// assign null
	d_ImgData = nullptr;
	imgData   = nullptr;
}