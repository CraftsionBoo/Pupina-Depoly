#ifndef __DEPTH_ANYTHING_V2_H__
#define __DEPTH_ANYTHING_V2_H__

#include <future>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "../../tensorRT/common/ilogger.hpp"
#include "../../tensorRT/common/cuda_tools.hpp"
#include "../../tensorRT/common/trt_tensor.hpp"

namespace depth
{
    using namespace Taurus;

    struct DepthEstimation
    {
        int x;
        int y;
        int label;

        DepthEstimation()
        {
            x = 0;
            y = 0;
            label = -1;
        }
    };
    void image_to_tensor(const cv::Mat &image, std::shared_ptr<TRT::Tensor> &tensor, int ibatch);

    class DepthInfer
    {
    public:
        virtual std::shared_future<cv::Mat> commit(const cv::Mat &image) = 0;
        virtual std::vector<std::shared_future<cv::Mat>> commits(const std::vector<cv::Mat> &images) = 0;
    };

    std::shared_ptr<DepthInfer> create_depth_infer(const std::string &engine_file, int gputid, bool use_multi_preprpcess_stream = false);
};

#endif