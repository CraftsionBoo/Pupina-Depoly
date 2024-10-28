#ifndef __YOLO_V5_HPP__
#define __YOLO_V5_HPP__

#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>

#include "tensorRT/common/ilogger.hpp"
#include "tensorRT/common/cuda_tools.hpp"

namespace v5
{
    using namespace Taurus;
    enum class NMSMethod : int
    {
        CPU = 0,
        FastGPU = 1
    };

    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };
    typedef std::vector<Box> BoxArray;

    class Infer
    {
    public:
        virtual std::shared_future<BoxArray> commit(const cv::Mat &image) = 0;
        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat> &images) = 0;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string &engine_file, int gpuid,
        float confidence_threshold = 0.25f, float nms_threshold = 0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false);

}; // namespace v5
#endif