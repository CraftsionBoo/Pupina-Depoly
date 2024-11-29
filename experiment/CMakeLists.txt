cmake_minimum_required(VERSION 3.10)
project(trtmodule LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(TensorRT_Sample_Dir /usr/src/tensorrt/samples)   
set(TensorRT_Sample_Include /usr/src/tensorrt/samples/common)
include_directories(/usr/local/cuda/include)

# 添加CMAKE_MODULE_PATH，否则找不到FindTensorRT.cmake
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(TensorRT REQUIRED)
if (TensorRT_FOUND)
    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
    message(STATUS "TensorRT include files: ${TensorRT_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Cannot find TensorRT")
endif()

# 添加可执行文件
add_executable(build 
    c/build.cpp
    ${TensorRT_Sample_Dir}/common/logger.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_Sample_Include})
target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES})
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")

add_executable(runtime 
    c/runtime.cu
    ${TensorRT_Sample_Dir}/common/logger.cpp
)
target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_Sample_Include})
target_link_libraries(runtime PRIVATE ${TensorRT_LIBRARIES})
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75")