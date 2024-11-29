#include "depth_v2.hpp"

#include "../../tensorRT/infer/trt_infer.hpp"
#include "../../tensorRT/infer/template_infer.hpp"
#include "../../tensorRT/common/preprocess_kernel.cuh"

namespace depth
{
    struct AffineMatrix // 仿射变换
    {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const cv::Size &from, const cv::Size &to)
        {
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;

            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat()
        {
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    using ControllerImpl = InferController<
        cv::Mat,                      // input
        cv::Mat,                      // output 1x
        std::tuple<std::string, int>, // start param
        AffineMatrix                  // additional
        >;
    class InferImpl : public DepthInfer, public ControllerImpl
    {
    public:
        virtual ~InferImpl() { stop(); }

        virtual bool startup(const std::string &file, int gpuid, bool use_multi_preprpcess_stream)
        {
            float mean[3] = {123.675, 116.28, 103.53};
            float std[3] = {58.395, 57.12, 57.375};
            normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1.0f, CUDAKernel::ChannelType::None);
            use_multi_preprocess_stream_ = use_multi_preprpcess_stream;
            return ControllerImpl::startup(std::make_tuple(file, gpuid));
        }

        virtual void worker(std::promise<bool> &result) override
        {
            std::string file = std::get<0>(start_param_);
            int gpuid = std::get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);

            if (engine == nullptr)
            {
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();
            TRT::Tensor output_depth_device(TRT::DataType::Float32);

            int max_batch_size = engine->get_max_batch_size(); // 1，没有设置动态batch
            auto input = engine->tensor("input");              // input : 1 x 3 x 518 x 518
            auto depth_head_output = engine->tensor("output"); // output : 1 x 518 x 518

            input_width_ = input->size(3);
            input_height_ = input->size(2);
            tensor_allocator_ = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_ = engine->get_stream();
            gpu_ = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            output_depth_device.resize(max_batch_size, input_width_ * input_height_).to_gpu();

            std::vector<Job> fetch_jobs;
            while (get_jobs_and_wait(fetch_jobs, max_batch_size))
            {
                int infer_batch_size = fetch_jobs.size();
                INFOI("infer ----------> batch size : %d", infer_batch_size);
                input->resize_single_dim(0, infer_batch_size);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    auto &job = fetch_jobs[ibatch];
                    auto &mono = job.mono_tensor->data();

                    if (mono->get_stream() != stream_)
                    {
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                output_depth_device.to_gpu(false); // 数据移植在gpu上
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    float *image_based_output = depth_head_output->gpu<float>(ibatch);
                    float *output_depth_ptr = output_depth_device.gpu<float>(ibatch); 
                    checkCudaRuntime(cudaMemcpyAsync(output_depth_ptr, image_based_output,
                                                     input_height_ * input_width_ * sizeof(float),
                                                     cudaMemcpyDeviceToDevice, stream_));
                }

                output_depth_device.to_cpu();
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    float *parray = output_depth_device.cpu<float>(ibatch);
                    auto &job = fetch_jobs[ibatch];
                    auto &depth_image = job.output;
                    depth_image.create(input_height_, input_width_, CV_32FC1);
                    memcpy(depth_image.data, parray, input_height_ * input_width_ * sizeof(float));

                    job.pro->set_value(depth_image);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFOI("Engine destory.");
        }

        virtual bool preprocess(Job &job, const cv::Mat &image) override
        {
            if (tensor_allocator_ == nullptr)
            {
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            if (image.empty())
            {
                INFOE("Image is empty");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if (job.mono_tensor == nullptr)
            {
                INFOE("Tensor allocatory query failed");
                return false;
            }

            Taurus::cudaTools::AutoDevice auto_device(gpu_);
            auto &tensor = job.mono_tensor->data();
            Taurus::TRT::CUStream preprocess_stream = nullptr;

            if (tensor == nullptr)
            {
                // not init
                tensor = std::make_shared<Taurus::TRT::Tensor>();
                tensor->set_workspace(std::make_shared<Taurus::TRT::Memory>());

                if (use_multi_preprocess_stream_)
                {
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));
                    tensor->set_stream(preprocess_stream, true);
                }
                else
                {
                    preprocess_stream = stream_;
                    tensor->set_stream(preprocess_stream, false);
                }
            }

            cv::Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);
            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            size_t size_matrix = Taurus::cUtils::upbound(sizeof(job.additional.d2i), 32);
            auto workspace = tensor->get_workspace();
            uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_matrix + size_image);
            float *affine_matrix_device = (float *)gpu_workspace;
            uint8_t *image_device = size_matrix + gpu_workspace;

            uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(size_matrix + size_image);
            float *affine_matrix_host = (float *)cpu_workspace;
            uint8_t *image_host = size_matrix + cpu_workspace;

            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, preprocess_stream));
            Taurus::CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device, image.cols * 3, image.cols, image.rows,
                tensor->gpu<float>(), input_width_, input_height_,
                affine_matrix_device, 114, normalize_, preprocess_stream);
            return true;
        }

        virtual std::vector<std::shared_future<cv::Mat>> commits(const std::vector<cv::Mat> &images) override
        {
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<cv::Mat> commit(const cv::Mat &image) override
        {
            return ControllerImpl::commit(image);
        }

    private:
        int gpu_ = 0;
        int input_width_ = 0;
        int input_height_ = 0;
        TRT::CUStream stream_ = nullptr;
        CUDAKernel::Norm normalize_;
        bool use_multi_preprocess_stream_ = false;
    };

    std::shared_ptr<DepthInfer> create_depth_infer(const std::string &engine_file, int gpuid,
                                                   bool use_multi_preprocess_stream)
    {
        std::shared_ptr<InferImpl> instance(new InferImpl());
        if (!instance->startup(engine_file, gpuid, use_multi_preprocess_stream))
        {
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat &image, std::shared_ptr<TRT::Tensor> &tensor, int ibatch)
    {
        CUDAKernel::Norm normalize;

        normalize = CUDAKernel::Norm::alpha_bata(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);

        cv::Size input_size(tensor->size(3), tensor->size(2));
        AffineMatrix affine;
        affine.compute(image.size(), input_size);

        size_t size_image = image.cols * image.rows * 3;
        size_t size_matrix = cUtils::upbound(sizeof(affine.d2i), 32);
        auto workspace = tensor->get_workspace();
        uint8_t *gpu_workspace = (uint8_t *)workspace->gpu(size_matrix + size_image);
        float *affine_matrix_device = (float *)gpu_workspace;
        uint8_t *image_device = size_matrix + gpu_workspace;

        uint8_t *cpu_workspace = (uint8_t *)workspace->cpu(size_matrix + size_image);
        float *affine_matrix_host = (float *)cpu_workspace;
        uint8_t *image_host = size_matrix + cpu_workspace;
        auto stream = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));

        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

        CUDAKernel::warp_affine_bilinear_and_normalize_plane(
            image_device, image.cols * 3, image.cols, image.rows,
            tensor->gpu<float>(ibatch), input_size.width, input_size.height,
            affine_matrix_device, 114, normalize, stream);
        tensor->synchronize();
    }
};
