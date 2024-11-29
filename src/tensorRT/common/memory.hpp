#ifndef __TRT_MEMORY_HPP__
#define __TRT_MEMORY_HPP__

#include <stddef.h>

#define CURRENT_DEVICE_ID -1

namespace Taurus
{
    namespace TRT
    {
        /**
         * 内存复用 多设备自动切换
         * **/
        class Memory
        {
        public:
            Memory(int device_id = CURRENT_DEVICE_ID);
            Memory(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id = CURRENT_DEVICE_ID);
            virtual ~Memory();

            inline size_t cpu_size() const { return cpu_size_; }
            inline size_t gpu_size() const { return gpu_size_; }
            inline bool owner_cpu() const { return owner_cpu_; }
            inline bool owner_gpu() const { return owner_gpu_; }
            inline int device_id() const { return device_id_; }

            inline void *cpu() const { return cpu_; }
            inline void *gpu() const { return gpu_; }

            void* cpu(size_t size);
            void* gpu(size_t size);

            void release_gpu();
            void release_cpu();
            void release_all();
            void reference_data(void *cpu, size_t cpu_size, void *gpu, size_t gpu_size, int device_id = CURRENT_DEVICE_ID);

        private:
            void *cpu_ = nullptr;
            size_t cpu_size_ = 0;
            bool owner_cpu_ = true;
            void *gpu_ = nullptr;
            size_t gpu_size_ = 0;
            bool owner_gpu_ = true;
            int device_id_ = 0;
        };

    }; // namespace TRT

}; // namespace Taurus

#endif