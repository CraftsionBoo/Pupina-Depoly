#ifndef __CUDA_TOOLS_HPP__
#define __CUDA_TOOLS_HPP__

#include <cuda_runtime.h>
#include <cuda.h>
#include <string>
#include <stdarg.h>
#include "ilogger.hpp"
#include "cv_cpp_utils.hpp"

#define GPU_BLOCK_THREADS 512

#define checkCudaDriver(call) Taurus::cudaTools::__check_driver(call, #call, __LINE__, __FILE__)
#define checkCudaRuntime(call) Taurus::cudaTools::__check_runtime(call, #call, __LINE__, __FILE__)
#define checkCUDAKernel(...)                                                 \
    __VA_ARGS__;                                                             \
    do                                                                       \
    {                                                                        \
        cudaError_t cuda_status = cudaPeekAtLastError();                     \
        if (cuda_status != cudaSuccess)                                      \
        {                                                                    \
            INFOE("Failed to Launch : %s", cudaGetErrorString(cuda_status)); \
        }                                                                    \
    } while (0);
namespace Taurus
{
    namespace cudaTools
    {
        bool __check_driver(CUresult e, const char *call, int line, const char *file);
        bool __check_runtime(cudaError_t e, const char *call, int line, const char *szfile);
        bool check_device_id(int device_id);

        class AutoDevice // 切换设备
        {
        public:
            AutoDevice(int device_id = 0);
            virtual ~AutoDevice(); // 返回默认

        private:
            int old = -1; // default
        };

        std::string description();
        int current_device_id();
        void display_current_useable_device();

        dim3 grid_dims(int numJobs);
        dim3 block_dims(int numJobs);
    }; // namesapce cudaTools
}; // namesapce Taurus

#endif