cmake_minimum_required(VERSION 3.22)

project(cudaFiles LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60)

add_subdirectory(src)
add_subdirectory(externals)

target_include_directories(
	${PROJECT_NAME}
	PUBLIC externals/
)

target_link_libraries(
	${PROJECT_NAME} 
	stbImage
)

set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
