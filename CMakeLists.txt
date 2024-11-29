cmake_minimum_required(VERSION 3.10)
project(tensorRT_taurus LANGUAGES CXX CUDA)

# 添加CMAKE_MODULE_PATH，否则找不到FindTensorRT.cmake
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)

list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
if (TensorRT_FOUND)
    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
    message(STATUS "TensorRT include files: ${TensorRT_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Cannot find TensorRT")
endif()

find_package(OpenCV 4.4.0 REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/application
    ${PROJECT_SOURCE_DIR}/src/tensorRT 
    ${PROJECT_SOURCE_DIR}/src/tensorRT/common
    ${PROJECT_SOURCE_DIR}/src/tensorRT/infer
    ${PROJECT_SOURCE_DIR}/src/tensorRT/builder
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

set(TAURUS_TENSORRT_DIR ${PROJECT_SOURCE_DIR}/src/tensorRT)
set(TAURUS_APPLICATION_DIR ${PROJECT_SOURCE_DIR}/src/application)

file(GLOB_RECURSE TAURUS_BASIC_FILES ${TAURUS_TENSORRT_DIR}/*.cpp ${TAURUS_TENSORRT_DIR}/*.cu)
add_library(taurus_basic SHARED ${TAURUS_BASIC_FILES})
target_link_libraries(taurus_basic ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(taurus_basic pthread cuda)

#  main
file(GLOB_RECURSE main_srcs ${TAURUS_APPLICATION_DIR}/*.cpp ${TAURUS_APPLICATION_DIR}/*.cu)
add_executable(taurus ${main_srcs}  ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(taurus ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(taurus ${CUDA_LIBRARIES} taurus_basic)
