#ifndef __TRT_TENSOR_HPP__
#define __TRT_TENSOR_HPP__

#include <memory>
#include <vector>

#include "memory.hpp"

struct CUstream_st;
typedef CUstream_st CUStreamRaw;

namespace Taurus
{
    namespace TRT
    {
        typedef struct
        {
            unsigned short _;
        } float16; // fp16
        typedef CUStreamRaw *CUStream;

        enum class DataHead : int // 设备id
        {
            Init = 0,
            Device = 1,
            Host = 2
        };

        enum class DataType : int // 数据类型
        {
            Unkown = -1,
            Float32 = 0,
            Float16 = 1, // 待定
            UInt8 = 2
        };

        /** utils **/
        const char* data_type_string(DataType dt);
        const char* data_head_string(DataHead dh);
        int data_type_size(DataType dt);

        class Tensor
        {
        public:
            Tensor(const Tensor &other) = delete;
            Tensor &operator=(const Tensor &other) = delete;

            explicit Tensor(int ndims, const int *dims, DataType dtype = DataType::Float32, std::shared_ptr<Memory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
            explicit Tensor(DataType dtype = DataType::Float32, std::shared_ptr<Memory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
            virtual ~Tensor();

            int count(int start_axis = 0) const;                                                  // return axis对应size
            int numel() const;                                                                    // return bs*c*h*w
            inline int ndims() const { return shape_.size(); }                                    // return 维度大小
            inline int size(int index) const { return shape_[index]; }                            // return bs c h w
            inline int shape(int index) const { return shape_[index]; }                           // =size
            inline int batch() const { return shape_[0]; }                                        // return bs
            inline int channel() const { return shape_[1]; }                                      // return channel
            inline int height() const { return shape_[2]; }                                       // return h
            inline int width() const { return shape_[3]; }                                        // return w
            inline DataType type() const { return dtype_; }                                       // return F32 F16 IN32 IN8
            inline const std::vector<int> &dims() const { return shape_; }                        // return [bs,c,h,w]
            inline const std::vector<size_t> &strides() const { return strides_; }                // return
            inline int btyes() const { return bytes_; }                                           // return bs*c*h*w*sizeof(DataType)
            inline int bytes(int start_axis) const { return count(start_axis) * element_size(); } // return bs/bs*c/bs*c*h/bs*c*h*w * sizeof(DataType)
            inline int element_size() const { return data_type_size(dtype_); }                    // return 数据类型内存大小
            inline DataHead head() { return head_; }                                              // return data current in host or device
            int device() const { return device_id_; }                                             // return device id
            std::shared_ptr<Memory> get_data() const { return data_; }
            std::shared_ptr<Memory> get_workspace() const { return workspace_; }
            CUStream get_stream() const { return stream_; }

            /** data access**/
            inline void *cpu() const
            {
                ((Tensor *)this)->to_cpu();
                return data_->cpu();
            }
            inline void *gpu() const
            {
                ((Tensor *)this)->to_gpu();
                return data_->gpu();
            }

            Tensor &to_cpu(bool copy = true); // 数据加载到cpu
            Tensor &to_gpu(bool copy = true); // 数据加载到gpu
            Tensor &to_half();                // 数据转换为f16
            Tensor &to_float();               // 数据转换为f32

            template <typename _DT>
            inline const _DT *cpu() const { return (_DT *)cpu(); }
            template <typename _DT>
            inline _DT *cpu() { return (_DT *)cpu(); }
            template <typename _DT>
            inline const _DT *gpu() const { return (_DT *)gpu(); }
            template <typename _DT>
            inline _DT *gpu() { return (_DT *)gpu(); }

            template <typename... _Args>
            int offset(int index, _Args... index_args) const
            {
                const int index_array[] = {index, index_args...};
                return offset_array(sizeof...(index_args) + 1, index_array);
            }
            int offset_array(const std::vector<int> &index) const;
            int offset_array(size_t size, const int *index_array) const;

            template <typename _DT, typename... _Args>
            inline _DT *cpu(int i, _Args &&...args) { return cpu<_DT>() + offset(i, args...); }
            template <typename _DT, typename... _Args>
            inline _DT *gpu(int i, _Args &&...args) { return gpu<_DT>() + offset(i, args...); }

            Tensor &set_stream(CUStream stream, bool owner = false)
            {
                stream_ = stream;
                stream_owner_ = owner;
                return *this;
            }
            Tensor &set_workspace(std::shared_ptr<Memory> workspace)
            {
                workspace_ = workspace;
                return *this;
            }

            template <typename... _Args>
            Tensor &resize(int dim_size, _Args... dim_size_args)
            {
                const int dim_size_array[] = {dim_size, dim_size_args...};
                return resize(sizeof...(dim_size_args) + 1, dim_size_array);
            }

            Tensor &resize(int ndims, const int *dims);
            Tensor &resize(const std::vector<int> &dims);
            Tensor &resize_single_dim(int idim, int size);
            Tensor &release();
            Tensor &synchronize();
            const char* shape_string() const {return shape_string_;}

            Tensor& copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id = CURRENT_DEVICE_ID);
            Tensor& copy_from_cpu(size_t offset, const void *src, size_t num_element);

        private:
            void setup_data(std::shared_ptr<Memory> data);
            Tensor &adajust_memory_by_update_dims_or_type();
            Tensor &compute_shape_string();

        private:
            CUStream stream_ = nullptr;
            bool stream_owner_ = false;
            std::shared_ptr<Memory> workspace_;
            std::vector<int> shape_;
            std::vector<size_t> strides_;
            size_t bytes_;
            DataHead head_ = DataHead::Init;
            DataType dtype_ = DataType::Float32;
            int device_id_ = 0;
            char shape_string_[100];
            char descriptor_string_[100];
            std::shared_ptr<Memory> data_;
        }; // class Tensor

    }; // namespace TRT
}; // namespace Taurus

#endif