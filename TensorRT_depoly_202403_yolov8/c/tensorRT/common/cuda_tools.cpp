#include "cuda_tools.hpp"

namespace Taurus
{
    namespace cudaTools
    {
        bool __check_driver(CUresult e, const char *call, int line, const char *file)
        {
            if (e != cudaSuccess)
            {
                const char *message = nullptr;
                const char *name = nullptr;
                cuGetErrorString(e, &message);
                cuGetErrorName(e, &name);
                INFOE("CUDA Driver error %s # %s, code = %s [ %d ] in file %s:%d", call, message, name, e, file, line);
                return false;
            }
            return true;
        }

        bool __check_runtime(cudaError_t e, const char *call, int line, const char *szfile)
        {
            if (e != cudaSuccess)
            {
                INFOE("CUDA runtime error %s # %s, code=%s [ %d ] in file %s:%d",
                      call,
                      cudaGetErrorString(e),
                      cudaGetErrorString(e),
                      e, szfile, line);
                return false;
            }
            return true;
        }

        bool check_device_id(int device_id)
        {
            int device_count = -1;
            checkCudaRuntime(cudaGetDeviceCount(&device_count));
            if (device_id < 0 || device_id >= device_count)
            {
                INFOE("Invalid device id: %d, count = %d", device_id, device_count);
                return false;
            }
            return true;
        }

        int current_device_id()
        {
            int device_id = 0;
            checkCudaRuntime(cudaGetDevice(&device_id));
            return device_id;
        }

        std::string description()
        {
            cudaDeviceProp prop;
            size_t free_mem, total_mem;
            int device_id = 0;

            checkCudaRuntime(cudaGetDevice(&device_id));
            checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
            checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));

            return Taurus::cUtils::format2048(
                "[ID %d]<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
                device_id, prop.name, prop.major, prop.minor,
                free_mem / 1024.0f / 1024.0f / 1024.0f,
                total_mem / 1024.0f / 1024.0f / 1024.0f);
        }

        void display_current_useable_device()
        {
            int device_nums = 0;
            checkCudaRuntime(cudaGetDeviceCount(&device_nums));
            if (device_nums == 0)
            {
                INFOI("current no device supporting CUDA");
            }
            else
            {
                INFOI("current %d device supporting CUDA");
                for (int i = 0; i < device_nums; ++i)
                {
                    cudaDeviceProp prop;
                    size_t free_mem, total_mem;
                    checkCudaRuntime(cudaGetDeviceProperties(&prop, i));
                    checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));

                    std::string device_info = Taurus::cUtils::format2048(
                        "<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
                        prop.name, prop.major, prop.minor,
                        free_mem / 1024.0f / 1024.0f / 1024.0f,
                        total_mem / 1024.0f / 1024.0f / 1024.0f);
                    INFOI("device %d: %s", i, device_info.c_str());
                }
            }
        }

        AutoDevice::AutoDevice(int device_id)
        {
            cudaGetDevice(&old);
            checkCudaRuntime(cudaSetDevice(device_id));
        }

        AutoDevice::~AutoDevice()
        {
            checkCudaRuntime(cudaSetDevice(old));
        }

        dim3 grid_dims(int numJobs)
        {
            int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
            return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
        }

        dim3 block_dims(int numJobs)
        {
            return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        }
    };
}; // namespace Taurus