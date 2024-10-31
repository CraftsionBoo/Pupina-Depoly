cmake_minimum_required(VERSION 3.10)
project(tensorRT_ultralytics LANGUAGES CXX CUDA)

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

find_package(OpenCV 3.2.0 REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

set(TAURUSTENSORRT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/c/tensorRT)

# compile common libs
file(GLOB TAURUS_COMMON_FILES ${TAURUSTENSORRT_DIR}/common/*.cpp ${TAURUSTENSORRT_DIR}/common/*.cu)
add_library(utensorrt_common SHARED ${TAURUS_COMMON_FILES})
target_link_libraries(utensorrt_common ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(utensorrt_common pthread cuda)

# compile builder libs
file(GLOB TAURUS_BUILDER_FILES ${TAURUSTENSORRT_DIR}/builder/*.cpp )
add_library(utensorrt_builder SHARED ${TAURUS_BUILDER_FILES})
target_link_libraries(utensorrt_builder ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(utensorrt_builder cuda utensorrt_common)

# compile Infer libs
file(GLOB TAURUS_INFER_FILES ${TAURUSTENSORRT_DIR}/infer/*.cpp )
add_library(utensorrt_infer SHARED ${TAURUS_INFER_FILES})
target_link_libraries(utensorrt_infer ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(utensorrt_infer cuda utensorrt_common)

# v8
set(V8PLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/c)
file(GLOB V8PLUGIN_DIR ${V8PLUGIN_DIR}/*cpp ${V8PLUGIN_DIR}/*.cu)
add_executable(app_v8_water_fish_seg_fp16 ${V8PLUGIN_DIR})
target_include_directories(app_v8_water_fish_seg_fp16 PRIVATE ${TensorRT_INCLUDE_DIRS})
target_link_libraries(app_v8_water_fish_seg_fp16 PRIVATE ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS} utensorrt_common utensorrt_builder utensorrt_infer)
