#ifndef __TRT_BUILDER_HPP__
#define __TRT_BUILDER_HPP__

#include <string>
#include <vector>
#include <functional>
#include "../common/trt_tensor.hpp"

namespace Taurus
{
    namespace TRT
    {
        enum class Mode : int
        {
            FP32,
            FP16,
            INT8
        };

        const char *mode_string(Mode type);

        typedef std::function<void(int current, int count, const std::vector<std::string> &files, std::shared_ptr<Tensor> &tensor)> Int8Preprocess; // 量化数据预处理

        bool compile(
            Mode mode,
            unsigned int maxBatchSize,
            const std::string &source,
            const std::string &saveto,
            const size_t maxWorkSpaceSize = 1ul << 28,
            Int8Preprocess int8process = nullptr,
            const std::string &int8ImageDirectory = "",
            const std::string &int8EntoryCalibratorFile = "");
    }; // namespace TRT
}; // namespace Taurus

#endif