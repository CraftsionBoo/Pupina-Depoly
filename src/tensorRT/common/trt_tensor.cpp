#include "trt_tensor.hpp"
#include "ilogger.hpp"
#include "cuda_tools.hpp"

#include <assert.h>

namespace Taurus
{
    namespace TRT
    {
        int data_type_size(DataType dt)
        {
            switch (dt)
            {
            case DataType::Float32:
                return sizeof(float);
            case DataType::Float16:
                return sizeof(float16);
            case DataType::UInt8:
                return sizeof(uint8_t);
            default:
            {
                INFOE("Not support dtype: %d", dt);
                return -1;
            }
            }
        }

        const char *data_type_string(DataType dt)
        {
            switch (dt)
            {
            case DataType::Float16:
                return "Float16";
            case DataType::Float32:
                return "Float";
            case DataType::UInt8:
                return "Uint8";
            default:
                return "Unknow";
            }
        }

        const char *data_head_string(DataHead dh)
        {
            switch (dh)
            {
            case DataHead::Init:
                return "Init";
            case DataHead::Device:
                return "Device";
            case DataHead::Host:
                return "Host";
            default:
                return "Unknow";
            }
        }

        //////////////////////////////////////////////////////////////////////////

        inline static int get_device(int device_id)
        {
            if (device_id != CURRENT_DEVICE_ID)
            {
                Taurus::cudaTools::check_device_id(device_id);
                return device_id;
            }

            checkCudaRuntime(cudaGetDevice(&device_id));
            return device_id;
        }

        Tensor::Tensor(int ndims, const int *dims, DataType dtype, std::shared_ptr<Memory> data, int device_id)
        {
            this->dtype_ = dtype;
            this->device_id_ = get_device(device_id);
            descriptor_string_[0] = 0;
            setup_data(data);
            resize(ndims, dims);
        }

        Tensor::Tensor(DataType dtype, std::shared_ptr<Memory> data, int device_id)
        {
            shape_string_[0] = 0;
            descriptor_string_[0] = 0;
            this->device_id_ = get_device(device_id);
            this->dtype_ = dtype;
            setup_data(data);
        }

        Tensor::~Tensor()
        {
            release();
        }

        void Tensor::setup_data(std::shared_ptr<Memory> data)
        {
            data_ = data;
            if (data_ == nullptr)
                data_ = std::make_shared<Memory>(device_id_);
            else
                device_id_ = data_->device_id();
            head_ = DataHead::Init;
            if (data_->cpu())
                head_ = DataHead::Host;

            if (data_->gpu())
                head_ = DataHead::Device;
        }

        int Tensor::count(int start_axis) const
        {
            if (start_axis >= 0 && start_axis < shape_.size())
            {
                int size = 1;
                for (int i = start_axis; i < shape_.size(); ++i)
                    size *= shape_[i];
                return size;
            }
            else
                return 0;
        }

        int Tensor::numel() const
        {
            int value = shape_.empty() ? 0 : 1;
            for (int i = 0; i < shape_.size(); ++i)
            {
                value *= shape_[i];
            }
            return value;
        }

        Tensor &Tensor::to_cpu(bool copy)
        {
            if (head_ == DataHead::Host)
                return *this;

            head_ = DataHead::Host;
            data_->cpu(bytes_); // 分配内存

            if (copy && data_->gpu() != nullptr) // gpu数据直接加载到cpu即可
            {
                Taurus::cudaTools::AutoDevice auto_e_device(this->device_id_);
                checkCudaRuntime(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
                checkCudaRuntime(cudaStreamSynchronize(stream_));
            }
            return *this;
        }

        Tensor &Tensor::to_gpu(bool copy)
        {
            if (head_ == DataHead::Device)
                return *this;

            head_ = DataHead::Device;
            data_->gpu(bytes_);

            if (copy && data_->cpu() != nullptr)
            {
                Taurus::cudaTools::AutoDevice auto_e_device(this->device_id_);
                checkCudaRuntime(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
            }
            return *this;
        }

        int Tensor::offset_array(size_t size, const int *index_array) const
        {
            assert(size <= shape_.size());
            int value = 0;
            for (int i = 0; i < shape_.size(); ++i)
            {
                if (i < size)
                    value += index_array[i];
                if (i + 1 < shape_.size())
                    value *= shape_[i + 1];
            }
            return value;
        }

        int Tensor::offset_array(const std::vector<int> &index) const
        {
            return offset_array(index.size(), index.data());
        }

        Tensor &Tensor::resize(const std::vector<int> &dims)
        {
            return resize(dims.size(), dims.data());
        }

        Tensor &Tensor::resize(int ndims, const int *dims)
        {
            std::vector<int> setup_dims(ndims);
            for (int i = 0; i < ndims; ++i)
            {
                int dim = dims[i];
                if (dim == -1) // bs == -1
                {
                    assert(ndims == shape_.size());
                    dim = shape_[i];
                }
                setup_dims[i] = dim;
            }
            this->shape_ = setup_dims;
            this->strides_.resize(setup_dims.size());

            size_t prev_size = element_size();
            size_t prev_shape = 1;
            for (int i = (int)strides_.size() - 1; i >= 0; --i)
            {
                if (i + 1 < strides_.size())
                {
                    prev_size = strides_[i + 1];
                    prev_shape = shape_[i + 1];
                }
                strides_[i] = prev_size * prev_shape;
            }
            this->adajust_memory_by_update_dims_or_type();
            this->compute_shape_string();
            return *this;
        }

        Tensor &Tensor::resize_single_dim(int idim, int size)
        {
            assert(idim >= 0 && idim < shape_.size());
            auto new_shape = shape_;
            new_shape[idim] = size;
            return resize(new_shape);
        }

        Tensor &Tensor::adajust_memory_by_update_dims_or_type() // 更新需要内存空间
        {
            int needed_size = this->numel() * element_size();
            if (needed_size > this->bytes_)
            {
                head_ = DataHead::Init;
            }
            this->bytes_ = needed_size;
            return *this;
        }

        Tensor &Tensor::compute_shape_string() // 打印输出
        {
            shape_string_[0] = 0;
            char *buffer = shape_string_;
            size_t buffer_size = sizeof(shape_string_);
            for (int i = 0; i < shape_.size(); ++i)
            {

                int size = 0;
                if (i < shape_.size() - 1)
                    size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
                else
                    size = snprintf(buffer, buffer_size, "%d", shape_[i]);

                buffer += size;
                buffer_size -= size;
            }
            return *this;
        }

        Tensor &Tensor::release()
        {
            data_->release_all();
            shape_.clear();
            bytes_ = 0;
            head_ = DataHead::Init;

            if (stream_owner_ && stream_ != nullptr)
            {
                Taurus::cudaTools::AutoDevice auto_e_device(this->device());
                checkCudaRuntime(cudaStreamDestroy(stream_));
            }
            stream_owner_ = false;
            stream_ = nullptr;
            return *this;
        }

        Tensor &Tensor::synchronize()
        {
            cudaTools::AutoDevice auto_e_device(this->device());
            checkCudaRuntime(cudaStreamSynchronize(stream_));
            return *this;
        }

        Tensor &Tensor::copy_from_cpu(size_t offset, const void *src, size_t num_element)
        {
            if (head_ == DataHead::Init)
                to_cpu(false);

            size_t offset_location = offset * element_size();
            if (offset_location >= bytes_)
            {
                INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
                return *this;
            }

            size_t copyed_bytes = num_element * element_size();
            size_t remain_bytes = bytes_ - offset_location;
            if (copyed_bytes > remain_bytes)
            {
                INFOE("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
                return *this;
            }

            if (head_ == DataHead::Device)
            {
                Taurus::cudaTools::AutoDevice auto_device_exchange(this->device());
                checkCudaRuntime(cudaMemcpyAsync((char *)data_->gpu() + offset_location, src, copyed_bytes, cudaMemcpyHostToDevice, stream_));
            }
            else if (head_ == DataHead::Host)
            {
                memcpy((char *)data_->cpu() + offset_location, src, copyed_bytes);
            }
            else
            {
                INFOE("Unspoort head type %d", head_);
            }
            return *this;
        }

        Tensor &Tensor::copy_from_gpu(size_t offset, const void *src, size_t num_element, int device_id)
        {
            if (head_ == DataHead::Init)
                to_gpu(false);

            size_t offset_location = offset * element_size();
            if (offset_location >= bytes_)
            {
                INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
                return *this;
            }

            size_t copyed_bytes = num_element * element_size();
            size_t remain_bytes = bytes_ - offset_location;
            if (copyed_bytes > remain_bytes)
            {
                INFOE("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
                return *this;
            }

            if (head_ == DataHead::Device)
            {
                int current_device_id = get_device(device_id);
                int gpu_device_id = device();
                if (current_device_id != gpu_device_id)
                {
                    checkCudaRuntime(cudaMemcpyPeerAsync(gpu<unsigned char>() + offset_location, gpu_device_id, src, current_device_id, copyed_bytes, stream_));
                }
                else
                {
                    checkCudaRuntime(cudaMemcpyAsync(gpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToDevice, stream_));
                }
            }
            else if (head_ == DataHead::Host)
            {
                Taurus::cudaTools::AutoDevice auto_device_exchange(this->device());
                checkCudaRuntime(cudaMemcpyAsync(cpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToHost, stream_));
            }
            else
            {
                INFOE("Unsupport head type %d", head_);
            }
            return *this;
        }

    }; // namespace TRT
}; // namespace Taurus