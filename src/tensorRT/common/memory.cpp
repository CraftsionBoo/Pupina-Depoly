#include <string.h>
#include <assert.h>

#include "memory.hpp"
#include "cuda_tools.hpp"

inline static int check_and_trans_device_id(int device_id)
{
    if (device_id != CURRENT_DEVICE_ID)
    {
        Taurus::cudaTools::check_device_id(device_id);
        return device_id;
    }
    checkCudaRuntime(cudaGetDevice(&device_id));
    return device_id;
}

namespace Taurus
{
    namespace TRT
    {
        Memory::Memory(int device_id)
        {
            device_id_ = check_and_trans_device_id(device_id);
        }

        Memory::Memory(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id)
        {
            reference_data(cpu, cpu_size, gpu, gpu_size, device_id);
        }

        void Memory::reference_data(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id)
        {
            release_all();
            if (cpu == nullptr || cpu_size == 0)
            {
                cpu = nullptr;
                cpu_size = 0;
            }
            if (gpu == nullptr || gpu_size == 0)
            {
                gpu = nullptr;
                gpu_size = 0;
            }

            this->cpu_ = cpu;
            this->cpu_size_ = cpu_size;
            this->gpu_ = gpu;
            this->gpu_size_ = gpu_size;
            this->owner_cpu_ = !(cpu && cpu_size > 0);
            this->owner_gpu_ = !(gpu && gpu_size > 0);
            device_id_ = check_and_trans_device_id(device_id);
        }

        Memory::~Memory()
        {
            release_all();
        }

        void Memory::release_all()
        {
            release_cpu();
            release_gpu();
        }

        void Memory::release_cpu()
        {
            if (cpu_)
            {
                if (owner_cpu_)
                {
                    Taurus::cudaTools::AutoDevice auto_e_device(device_id_);
                    checkCudaRuntime(cudaFreeHost(cpu_));
                }
                cpu_ = nullptr;
            }
            cpu_size_ = 0;
        }

        void Memory::release_gpu()
        {
            if (gpu_)
            {
                if (owner_gpu_)
                {
                    Taurus::cudaTools::AutoDevice auto_e_device(device_id_);
                    checkCudaRuntime(cudaFree(gpu_));
                }
                gpu_ = nullptr;
            }
            gpu_size_ = 0;
        }

        void *Memory::cpu(size_t size)
        {
            if (cpu_size_ < size)
            {
                release_cpu();

                cpu_size_ = size;
                Taurus::cudaTools::AutoDevice auto_e_device(device_id_);
                checkCudaRuntime(cudaMallocHost(&cpu_, size));
                assert(cpu_ != nullptr);
                memset(cpu_, 0, size); // 初始化为0
            }
            return cpu_;
        }

        void *Memory::gpu(size_t size)
        {
            if (gpu_size_ < size)
            {
                release_gpu();
                gpu_size_ = size;
                Taurus::cudaTools::AutoDevice auto_e_device(device_id_);
                checkCudaRuntime(cudaMalloc(&gpu_, size));
                checkCudaRuntime(cudaMemset(gpu_, 0, size)); // 初始化为0
            }
            return gpu_;
        }

    }; // namespace TRT
}; // namespace Taurus