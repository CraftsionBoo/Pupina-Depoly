#include "yolo_seg.hpp"

#include "../../tensorRT/infer/trt_infer.hpp"
#include "../../tensorRT/infer/template_infer.hpp"
#include "../../tensorRT/common/preprocess_kernel.cuh"

namespace YoloSeg
{
    InstanceSegmentMap::InstanceSegmentMap(int _width, int _height)
    {
        this->width = _width;
        this->height = _height;
        checkCudaRuntime(cudaMallocHost(&this->data, _width * _height));
    }

    InstanceSegmentMap::~InstanceSegmentMap()
    {
        if (this->data)
        {
            checkCudaRuntime(cudaFreeHost(this->data));
        }
        this->width = 0;
        this->height = 0;
    }

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

    static float iou(const Box &a, const Box &b)
    {
        float cleft = std::max(a.left, b.left);
        float ctop = std::max(a.top, b.top);
        float cright = std::min(a.right, b.right);
        float cbottom = std::min(a.bottom, b.bottom);

        float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
        if (c_area == 0.0f)
            return 0.0f;

        float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
        float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
        return c_area / (a_area + b_area - c_area);
    }

    static Boxseg cpu_nms(Boxseg &boxes, float threshold)
    {

        std::sort(boxes.begin(), boxes.end(), [](Boxseg::const_reference a, Boxseg::const_reference b)
                  { return a.confidence > b.confidence; });

        Boxseg output;
        output.reserve(boxes.size());

        std::vector<bool> remove_flags(boxes.size());
        for (int i = 0; i < boxes.size(); ++i)
        {

            if (remove_flags[i])
                continue;

            auto &a = boxes[i];
            output.emplace_back(a);

            for (int j = i + 1; j < boxes.size(); ++j)
            {
                if (remove_flags[j])
                    continue;

                auto &b = boxes[j];
                if (b.class_label == a.class_label)
                {
                    if (iou(a, b) >= threshold)
                        remove_flags[j] = true;
                }
            }
        }
        return output;
    }

    void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
                               float *invert_affine_matrix, float *parray, int max_objects, cudaStream_t stream);

    void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, cudaStream_t stream);

    static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
    {
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                            int mask_width, int mask_height, unsigned char *mask_out,
                            int mask_dim, int out_width, int out_height, cudaStream_t stream);

    using ControllerImpl = InferController<
        cv::Mat,                      // input
        Boxseg,                       // output
        std::tuple<std::string, int>, // start param
        AffineMatrix                  // additional
        >;
    class InferImpl : public SegInfer, public ControllerImpl
    {
    public:
        virtual ~InferImpl() { stop(); }

        virtual bool startup(
            const std::string &file, int gpuid,
            float confidence_threshold,
            float nms_threshold,
            NMSMethod nms_method,
            int max_objects,
            bool use_multi_preprpcess_stream)
        {
            normalize_ = CUDAKernel::Norm::alpha_bata(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            use_multi_preprocess_stream_ = use_multi_preprpcess_stream;
            confidence_threshold_ = confidence_threshold;
            nms_method_ = nms_method;
            max_objects_ = max_objects;
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
            const int MAX_IMAGE_BBOX = max_objects_;
            const int NUM_BOX_ELEMENT = 8; // left top right bottom confidence classes keepflag row_index(output)
            TRT::Tensor affine_matrix_device(TRT::DataType::Float32);
            TRT::Tensor output_boxarray_device(TRT::DataType::Float32);
            TRT::Tensor box_mask_output_memory(TRT::DataType::UInt8);

            int max_batch_size = engine->get_max_batch_size(); // max_bs = 10
            auto input = engine->tensor("images");             // 10 x 3 x 640 x 640
            auto bbox_head_output = engine->tensor("output0"); // detection head -1x8400x116
            auto mask_head_output = engine->tensor("output1"); // segment head -1x32x160x160
            int num_classes = bbox_head_output->size(2) - 4 - mask_head_output->size(1);

            input_width_ = input->size(3);
            input_height_ = input->size(2);
            tensor_allocator_ = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_ = engine->get_stream();
            gpu_ = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affine_matrix_device.set_stream(stream_);
            affine_matrix_device.resize(max_batch_size, 8).to_gpu(); // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_boxarray_device.resize(max_batch_size, 32 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

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

                    affine_matrix_device.copy_from_gpu(affine_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false); // 推理
                output_boxarray_device.to_gpu(false);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    auto &job = fetch_jobs[ibatch];
                    float *image_based_output = bbox_head_output->gpu<float>(ibatch);
                    float *output_array_ptr = output_boxarray_device.gpu<float>(ibatch);
                    auto affine_matrix = affine_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker(image_based_output, bbox_head_output->size(1), num_classes, confidence_threshold_,
                                          affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);
                    if (nms_method_ == NMSMethod::FastGPU)
                    {
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                }

                output_boxarray_device.to_cpu();
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch)
                {
                    float *parray = output_boxarray_device.cpu<float>(ibatch);
                    int count = std::min(MAX_IMAGE_BBOX, (int)*parray);
                    auto &job = fetch_jobs[ibatch];
                    auto &image_based_boxes = job.output;

                    for (int i = 0; i < count; ++i)
                    {
                        float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                        int keepflag = pbox[6];
                        if (keepflag == 1)
                        {
                            Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5]);

                            int row_index = pbox[7];
                            int mask_dim = mask_head_output->size(1);

                            float *mask_weights = bbox_head_output->gpu<float>(ibatch) + row_index * bbox_head_output->size(2) + num_classes + 4;
                            float *mask_head_predict = mask_head_output->gpu<float>(ibatch);
                            float left, top, right, bottom;
                            float *i2d = job.additional.i2d;
                            affine_project(i2d, pbox[0], pbox[1], &left, &top);
                            affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

                            float box_width = right - left;
                            float box_height = bottom - top;
                            float scale_to_predict_x = mask_head_output->size(3) / (float)input_width_;
                            float scale_to_predict_y = mask_head_output->size(2) / (float)input_height_;
                            int mask_out_width = box_width * scale_to_predict_x + 0.5f;
                            int mask_out_height = box_height * scale_to_predict_y + 0.5f;

                            if (mask_out_width > 0 && mask_out_height > 0)
                            {
                                int bytes_of_mask_out = mask_out_width * mask_out_height;
                                box_mask_output_memory.resize(bytes_of_mask_out).to_gpu();
                                box_mask_output_memory.to_gpu(false);
                                result_object_box.seg = std::make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);
                                unsigned char *mask_out_device = box_mask_output_memory.gpu<unsigned char>();
                                unsigned char *mask_out_host = result_object_box.seg->data;

                                decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                                                   mask_head_predict, mask_head_output->size(3), mask_head_output->size(2),
                                                   mask_out_device, mask_dim, mask_out_width, mask_out_height, stream_);
                                result_object_box.seg->left = left * scale_to_predict_x;
                                result_object_box.seg->top = top * scale_to_predict_y;
                                checkCudaRuntime(cudaMemcpyAsync(mask_out_host, mask_out_device, bytes_of_mask_out, cudaMemcpyDeviceToHost, stream_));
                                image_based_boxes.emplace_back(result_object_box);
                            }
                        }
                    }
                    if (nms_method_ == NMSMethod::CPU)
                    {
                        image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);
                    }
                    checkCudaRuntime(cudaStreamSynchronize(stream_));
                    job.pro->set_value(image_based_boxes);
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

        virtual std::vector<std::shared_future<Boxseg>> commits(const std::vector<cv::Mat> &images) override
        {
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<Boxseg> commit(const cv::Mat &image) override
        {
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_ = 0;
        int max_objects_ = 1024;
        NMSMethod nms_method_ = NMSMethod::FastGPU;
        TRT::CUStream stream_ = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    }; // class

    std::shared_ptr<SegInfer> create_seg_infer(
        const std::string &engine_file, int gpuid,
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects,
        bool use_multi_preprocess_stream)
    {
        std::shared_ptr<InferImpl> instance(new InferImpl);

        if (!instance->startup(engine_file, gpuid, confidence_threshold,
                               nms_threshold, nms_method, max_objects, use_multi_preprocess_stream))
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

}; // namespace YoloSeg