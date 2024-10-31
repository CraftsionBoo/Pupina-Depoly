#ifndef __YOLO_V8_HPP__
#define __YOLO_V8_HPP__

#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>

#include "tensorRT/common/ilogger.hpp"
#include "tensorRT/common/cuda_tools.hpp"

namespace v8
{
    using namespace Taurus;
    enum class NMSMethod : int
    {
        CPU = 0,
        FastGPU = 1
    };

    struct InstanceSegmentMap
    {
        int width = 0, height = 0;
        uint8_t *data = nullptr; // width * height memory

        InstanceSegmentMap(int _width, int _height);
        virtual ~InstanceSegmentMap();
    };

    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;
        std::shared_ptr<InstanceSegmentMap> seg;

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };

    typedef std::vector<Box> Boxseg;
    enum class Task : int
    {
        det = 0,
        seg = 1,
        clf = 2,
        pos = 3
    };

    class SegInfer
    {
    public:
        virtual std::shared_future<Boxseg> commit(const cv::Mat &image) = 0;
        virtual std::vector<std::shared_future<Boxseg>> commits(const std::vector<cv::Mat> &images) = 0;
    };

    std::shared_ptr<SegInfer> create_seg_infer(
        const std::string &engine_file, Task task = Task::seg, int gpuid = 0,
        float confidence_threshold = 0.25f, float nms_threshold = 0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false);
};
#endif