#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include "utils/utils.cuh"
#include "stb_image/stb_image_write.h"

__global__ void GradientImg(uint8_t* outImg, const int width, const int height, const int channels)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= width || y >= height)
		return;

	int idx = (x + y * width) * channels;

	float r = float(x) / float(width);
	float g = float(y) / float(height);
	float b = 0.8f;
	
	outImg[idx]     = uint8_t(r * 255.99f);
	outImg[idx + 1] = uint8_t(g * 255.99f);
	outImg[idx + 2] = uint8_t(b * 255.99f);
	outImg[idx + 3] = 255;
}

int main()
{
	// kernel properties
	constexpr int tx = 32;
	constexpr int ty = 32;
	// image properties
	constexpr float aspectRatio = 16.0f / 9.0f;
	constexpr int height        = 540;
	constexpr int width         = height * aspectRatio;
	constexpr int channels      = 4; // number of channels in the image
	constexpr uint64_t imgSize  = width * height * channels;
	const char* outPath = "img/gradientImg.png";

	// allocate host memory
	uint8_t* imgData = new uint8_t[imgSize];
	// allocate device memory
	uint8_t* d_ImgData = nullptr;
	cudaCheckError(cudaMalloc(&d_ImgData, imgSize * sizeof(uint8_t)));
	
	// calc block size and grid size
	const dim3 threads(tx, ty); // block size
	const dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty); // grid size
	// call cuda kernel
	printf("Generating gradient image...\n");
	GradientImg<<<blocks, threads>>>(d_ImgData, width, height, channels);
	cudaCheckError(cudaGetLastError());
	// copy image data from device memory to host memory
	cudaCheckError(cudaMemcpy(imgData, d_ImgData, imgSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	// write image
	stbi_flip_vertically_on_write(true);
	stbi_write_png(outPath, width, height, channels, imgData, width * channels);
	printf("Image exported to %s.\n", outPath);

	// free memory
	cudaCheckError(cudaFree(d_ImgData));
	delete[] imgData;

	d_ImgData = imgData = nullptr;
}
