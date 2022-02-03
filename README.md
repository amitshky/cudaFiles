# cudaFiles
Repo for CUDA files.
This repo has only been built and tested in Windows.

## Prerequisites
* [CMake](https://cmake.org/download/)
* [CUDA runtime library](https://developer.nvidia.com/cuda-downloads) (This repo was built and tested on v11.6)

## Configuration
Configure cmake build files.\
(Run these commands in the root directory of this repo.)
### For Windows
```
./config.bat
```

## Build and run
(Run these commands in the root directory of this repo.)
### For Windows
Build and run in Debug mode.
```
./build.bat
./run.bat
```

Build and run in Release mode.
```
./build-rel.bat
./run-rel.bat
```

## Output
Some screenshots of the outputs of the programs.

<!--* write image caption

<img src="img/image.png" width=192>-->

* Grayscale

<img src="img/grayscale.png" width=192>

## Resources
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [NVCC commands manual](https://helpmanual.io/help/nvcc/)
* [CUDA - Modern CMake](https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html)
* [Building Cross-Platform CUDA Applications with CMake](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)
* [CUDA by Example](https://developer.nvidia.com/cuda-example)
* [CUDA Memory model](https://www.3dgep.com/cuda-memory-model/)