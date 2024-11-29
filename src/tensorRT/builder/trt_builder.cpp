#include "trt_builder.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include <sstream>
#include <chrono>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <memory>
#include <fstream>

#include "../common/ilogger.hpp"
#include "../common/cuda_tools.hpp"

class Logger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity == Severity::kINTERNAL_ERROR)
        {
            INFOE("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        }
        else if (severity == Severity::kERROR)
        {
            INFOE("NVInfer: %s", msg);
        }
        else if (severity == Severity::kWARNING)
        {
            INFOW("Nvinfer: %s", msg);
        }
        else
            INFOD("%s", msg);
    }
};

static Logger glogger;

namespace Taurus
{
    namespace TRT
    {
        /***********************  functions  *************************/
        const char *mode_string(Mode type)
        {
            switch (type)
            {
            case Mode::FP32:
                return "FP32";
            case Mode::FP16:
                return "FP16";
            case Mode::INT8:
                return "INT8";
            default:
                return "UnknowTRTMode";
            }
        }

        // 智能指针模板
        template <typename _T>
        std::shared_ptr<_T> make_nvshared(_T *ptr)
        {
            return std::shared_ptr<_T>(ptr, [](_T *p)
                                       { p->destroy(); });
        }

        static std::string join_dims(const std::vector<int> &dims)
        {
            std::stringstream output;
            char buf[64];
            const char *fmts[] = {"%d", " x %d"};
            for (int i = 0; i < dims.size(); ++i)
            {
                snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
                output << buf;
            }
            return output.str();
        }

        bool save_file(const std::string &file, const void *data, size_t length)
        {
            FILE *f = fopen(file.c_str(), "wb");
            if (!f)
                return false;
            if (data && length > 0)
            {
                if (fwrite(data, 1, length, f) not_eq length) // not_eq 等价于!=
                {
                    fclose(f);
                    return false;
                }
            }
            fclose(f);
            return true;
        }

        static std::string format(const char *fmt, ...)
        {
            va_list vl;
            va_start(vl, fmt);
            char buffer[10000];
            vsprintf(buffer, fmt, vl);
            return buffer;
        }

        static std::string dims_str(const nvinfer1::Dims &dims)
        {
            return join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
        }

        // 采样
        static const char *padding_mode_name(nvinfer1::PaddingMode mode)
        {
            switch (mode)
            {
            case nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN:
                return "explicit round down";
            case nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP:
                return "explicit round up";
            case nvinfer1::PaddingMode::kSAME_UPPER:
                return "same supper";
            case nvinfer1::PaddingMode::kSAME_LOWER:
                return "same lower";
            case nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN:
                return "caffe round down";
            case nvinfer1::PaddingMode::kCAFFE_ROUND_UP:
                return "caffe round up";
            }
            return "Unknow padding mode";
        }

        // 池化层
        static const char *pooling_type_name(nvinfer1::PoolingType type)
        {
            switch (type)
            {
            case nvinfer1::PoolingType::kMAX:
                return "MaxPooling";
            case nvinfer1::PoolingType::kAVERAGE:
                return "AveragePooling";
            case nvinfer1::PoolingType::kMAX_AVERAGE_BLEND:
                return "MaxAverageBlendPooling";
            }
            return "Unknow pooling type";
        }

        // 激活层
        static const char *activation_type_name(nvinfer1::ActivationType activation_type)
        {
            switch (activation_type)
            {
            case nvinfer1::ActivationType::kRELU:
                return "ReLU";
            case nvinfer1::ActivationType::kSIGMOID:
                return "Sigmoid";
            case nvinfer1::ActivationType::kTANH:
                return "TanH";
            case nvinfer1::ActivationType::kLEAKY_RELU:
                return "LeakyRelu";
            case nvinfer1::ActivationType::kELU:
                return "Elu";
            case nvinfer1::ActivationType::kSELU:
                return "Selu";
            case nvinfer1::ActivationType::kSOFTSIGN:
                return "Softsign";
            case nvinfer1::ActivationType::kSOFTPLUS:
                return "Parametric softplus";
            case nvinfer1::ActivationType::kCLIP:
                return "Clip";
            case nvinfer1::ActivationType::kHARD_SIGMOID:
                return "Hard sigmoid";
            case nvinfer1::ActivationType::kSCALED_TANH:
                return "Scaled tanh";
            case nvinfer1::ActivationType::kTHRESHOLDED_RELU:
                return "Thresholded ReLU";
            }
            return "Unknow activation type";
        }

        // 网络层
        static std::string layer_type_name(nvinfer1::ILayer *layer)
        {
            switch (layer->getType())
            {
            case nvinfer1::LayerType::kCONVOLUTION:
                return "Convolution";
            case nvinfer1::LayerType::kFULLY_CONNECTED:
                return "Fully connected";
            case nvinfer1::LayerType::kACTIVATION:
            {
                nvinfer1::IActivationLayer *act = (nvinfer1::IActivationLayer *)layer;
                auto type = act->getActivationType();
                return activation_type_name(type);
            }
            case nvinfer1::LayerType::kPOOLING:
            {
                nvinfer1::IPoolingLayer *pool = (nvinfer1::IPoolingLayer *)layer;
                return pooling_type_name(pool->getPoolingType());
            }
            case nvinfer1::LayerType::kLRN:
                return "LRN";
            case nvinfer1::LayerType::kSCALE:
                return "Scale";
            case nvinfer1::LayerType::kSOFTMAX:
                return "SoftMax";
            case nvinfer1::LayerType::kDECONVOLUTION:
                return "Deconvolution";
            case nvinfer1::LayerType::kCONCATENATION:
                return "Concatenation";
            case nvinfer1::LayerType::kELEMENTWISE:
                return "Elementwise";
            case nvinfer1::LayerType::kPLUGIN:
                return "Plugin";
            case nvinfer1::LayerType::kUNARY:
                return "UnaryOp operation";
            case nvinfer1::LayerType::kPADDING:
                return "Padding";
            case nvinfer1::LayerType::kSHUFFLE:
                return "Shuffle";
            case nvinfer1::LayerType::kREDUCE:
                return "Reduce";
            case nvinfer1::LayerType::kTOPK:
                return "TopK";
            case nvinfer1::LayerType::kGATHER:
                return "Gather";
            case nvinfer1::LayerType::kMATRIX_MULTIPLY:
                return "Matrix multiply";
            case nvinfer1::LayerType::kRAGGED_SOFTMAX:
                return "Ragged softmax";
            case nvinfer1::LayerType::kCONSTANT:
                return "Constant";
            case nvinfer1::LayerType::kRNN_V2:
                return "RNNv2";
            case nvinfer1::LayerType::kIDENTITY:
                return "Identity";
            case nvinfer1::LayerType::kPLUGIN_V2:
                return "PluginV2";
            case nvinfer1::LayerType::kSLICE:
                return "Slice";
            case nvinfer1::LayerType::kSHAPE:
                return "Shape";
            case nvinfer1::LayerType::kPARAMETRIC_RELU:
                return "Parametric ReLU";
            case nvinfer1::LayerType::kRESIZE:
                return "Resize";
            }
            return "Unknow layer type";
        }

        static std::string layer_descript(nvinfer1::ILayer *layer)
        {
            switch (layer->getType())
            {
            case nvinfer1::LayerType::kCONVOLUTION:
            {
                nvinfer1::IConvolutionLayer *conv = (nvinfer1::IConvolutionLayer *)layer;
                return format("channel: %d, kernel: %s, padding: %s, stride: %s, dilation: %s, group: %d",
                              conv->getNbOutputMaps(),
                              dims_str(conv->getKernelSizeNd()).c_str(),
                              dims_str(conv->getPaddingNd()).c_str(),
                              dims_str(conv->getStrideNd()).c_str(),
                              dims_str(conv->getDilationNd()).c_str(),
                              conv->getNbGroups());
            }
            case nvinfer1::LayerType::kFULLY_CONNECTED:
            {
                nvinfer1::IFullyConnectedLayer *fully = (nvinfer1::IFullyConnectedLayer *)layer;
                return format("output channels: %d", fully->getNbOutputChannels());
            }
            case nvinfer1::LayerType::kPOOLING:
            {
                nvinfer1::IPoolingLayer *pool = (nvinfer1::IPoolingLayer *)layer;
                return format(
                    "window: %s, padding: %s",
                    dims_str(pool->getWindowSizeNd()).c_str(),
                    dims_str(pool->getPaddingNd()).c_str());
            }
            case nvinfer1::LayerType::kDECONVOLUTION:
            {
                nvinfer1::IDeconvolutionLayer *conv = (nvinfer1::IDeconvolutionLayer *)layer;
                return format("channel: %d, kernel: %s, padding: %s, stride: %s, group: %d",
                              conv->getNbOutputMaps(),
                              dims_str(conv->getKernelSizeNd()).c_str(),
                              dims_str(conv->getPaddingNd()).c_str(),
                              dims_str(conv->getStrideNd()).c_str(),
                              conv->getNbGroups());
            }
            case nvinfer1::LayerType::kACTIVATION:
            case nvinfer1::LayerType::kPLUGIN:
            case nvinfer1::LayerType::kLRN:
            case nvinfer1::LayerType::kSCALE:
            case nvinfer1::LayerType::kSOFTMAX:
            case nvinfer1::LayerType::kCONCATENATION:
            case nvinfer1::LayerType::kELEMENTWISE:
            case nvinfer1::LayerType::kUNARY:
            case nvinfer1::LayerType::kPADDING:
            case nvinfer1::LayerType::kSHUFFLE:
            case nvinfer1::LayerType::kREDUCE:
            case nvinfer1::LayerType::kTOPK:
            case nvinfer1::LayerType::kGATHER:
            case nvinfer1::LayerType::kMATRIX_MULTIPLY:
            case nvinfer1::LayerType::kRAGGED_SOFTMAX:
            case nvinfer1::LayerType::kCONSTANT:
            case nvinfer1::LayerType::kRNN_V2:
            case nvinfer1::LayerType::kIDENTITY:
            case nvinfer1::LayerType::kPLUGIN_V2:
            case nvinfer1::LayerType::kSLICE:
            case nvinfer1::LayerType::kSHAPE:
            case nvinfer1::LayerType::kPARAMETRIC_RELU:
            case nvinfer1::LayerType::kRESIZE:
                return "";
            }
            return "Unknow layer type";
        }

        static nvinfer1::Dims convert_to_trt_dims(const std::vector<int> &dims)
        {
            nvinfer1::Dims output{0};
            if (dims.size() > nvinfer1::Dims::MAX_DIMS)
            {
                INFOE("convert failed, dims.size[%d] > MAX_DIMS[%d]", dims.size(), nvinfer1::Dims::MAX_DIMS);
                return output;
            }

            if (dims.empty())
            {
                output.nbDims = dims.size();
                memcpy(output.d, dims.data(), dims.size() * sizeof(int));
            }
            return output;
        }

        static std::string align_blank(const std::string &input, int align_size, char blank = ' ')
        {
            if (input.size() >= align_size)
                return input;
            std::string output = input;
            for (int i = 0; i < align_size - input.size(); ++i)
                output.push_back(blank);
            return output;
        }

        static bool layer_has_input_tensor(nvinfer1::ILayer *layer) // 输入tensor显示
        {
            int num_input = layer->getNbInputs();
            for (int i = 0; i < num_input; ++i)
            {
                auto input = layer->getInput(i);
                if (input == nullptr)
                    continue;

                if (input->isNetworkInput())
                    return true;
            }
            return false;
        }

        static bool layer_has_output_tensor(nvinfer1::ILayer *layer) // 输出tensor显示
        {
            int num_output = layer->getNbOutputs();
            for (int i = 0; i < num_output; ++i)
            {
                auto output = layer->getOutput(i);
                if (output == nullptr)
                    continue;

                if (output->isNetworkOutput())
                    return true;
            }
            return false;
        }

        /***********************  INT8  *************************/
        /** entropy **/
        class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
        {
        private:
            std::vector<std::string> mImageNames; // 读取图像名
            size_t mBatchCudaSize = 0;
            int mCurrsor = 0;
            nvinfer1::Dims mDims;
            std::vector<std::string> mFiles;
            std::shared_ptr<Tensor> mTensor;
            std::vector<uint8_t> mEntropyCalibratorData;
            bool mFromCalibratorData = false;
            CUStream mStream = nullptr;
            Int8Preprocess mPreprocess;
            std::string mCacheFilename;

        public:
            /** 原始图像读入 **/
            Int8EntropyCalibrator(const std::vector<std::string> &img_files, nvinfer1::Dims dims, const Int8Preprocess &preprocess,const std::string cacheName)
            {
                assert(preprocess != nullptr);
                this->mDims = dims;
                this->mImageNames = img_files;
                this->mPreprocess = preprocess;
                this->mFromCalibratorData = false;
                this->mCacheFilename = cacheName;
                mFiles.resize(dims.d[0]);
                checkCudaRuntime(cudaStreamCreate(&mStream));
            }

            /** 缓存数据读入 **/
            Int8EntropyCalibrator(const std::vector<uint8_t> &entropyCalibratorData, nvinfer1::Dims dims, const Int8Preprocess &preprocess)
            {
                assert(preprocess != nullptr);
                this->mDims = dims;
                this->mEntropyCalibratorData = entropyCalibratorData;
                this->mPreprocess = preprocess;
                this->mFromCalibratorData = true;
                mFiles.resize(dims.d[0]);
                checkCudaRuntime(cudaStreamCreate(&mStream));
            }

            ~Int8EntropyCalibrator()
            {
                checkCudaRuntime(cudaStreamDestroy(mStream));
            }

            int getBatchSize() const noexcept
            {
                return mDims.d[0];
            }

            bool next() // 加载
            {
                INFOI("mdims : %d", mDims.d[0]);
                int batch_size = mDims.d[0];
                if (mCurrsor + batch_size > mImageNames.size())
                    return false;

                for (int i = 0; i < batch_size; ++i)
                    mFiles[i] = mImageNames[mCurrsor++];

                if (!mTensor)
                {
                    mTensor.reset(new Tensor(mDims.nbDims, mDims.d));
                    mTensor->set_stream(mStream); // 异步并行
                    mTensor->set_workspace(std::make_shared<Memory>());
                }
                INFOI("mdims : %d %d", mDims.d[1], mDims.d[2]);
                mPreprocess(mCurrsor, mImageNames.size(), mFiles, mTensor);
                return true;
            }

            bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept
            {
                if (!next())
                    return false;
                bindings[0] = mTensor->gpu(); // 数据加载到gpu
                return true;
            }

            const std::vector<uint8_t> &getEntropyCalibratorData()
            {
                return mEntropyCalibratorData;
            }

            const void *readCalibrationCache(size_t &length) noexcept
            {
                if (mFromCalibratorData)
                {
                    length = this->mEntropyCalibratorData.size();
                    return this->mEntropyCalibratorData.data();
                }
                length = 0;
                return nullptr;
            }

            virtual void writeCalibrationCache(const void *cache, size_t length) noexcept
            {
                // mEntropyCalibratorData.assign((uint8_t *)cache, (uint8_t *)cache + length);
                std::ofstream outCache(mCacheFilename, std::ios::binary);
                outCache.write(reinterpret_cast<const char*>(cache), length);
            }
        };

        bool compile(
            Mode mode,
            unsigned int maxBatchSize,
            const std::string &source,
            const std::string &saveto,
            const size_t maxWorkSpaceSize,
            Int8Preprocess int8process,
            const std::string &int8ImageDirectory,
            const std::string &int8EntoryCalibratorFile)
        {
            bool hasEntropyCalibrator = false;
            std::vector<uint8_t> entropyCalibratorData;
            std::vector<std::string> entropyCalibratorFiles;

            if (mode == Mode::INT8)
            {
                if (!int8EntoryCalibratorFile.empty())
                {
                    if (Taurus::cUtils::exists(int8EntoryCalibratorFile))
                    {
                        entropyCalibratorData = Taurus::cUtils::load_file(int8EntoryCalibratorFile);
                        if (entropyCalibratorData.empty())
                        {
                            INFOE("entropyCalibratorFile is set as %s, but we read is empty", int8EntoryCalibratorFile.c_str());
                            return false;
                        }
                        hasEntropyCalibrator = true;
                    }
                }

                if (hasEntropyCalibrator)
                {
                    if (!int8ImageDirectory.empty())
                    {
                        INFOW("image directory is ignore, when entropyCalibrator is set");
                    }
                }
                else
                {
                    if (int8process == nullptr)
                    {
                        INFOE("int8preprocess must be set, when mode is '%s'", mode_string(mode));
                        return false;
                    }

                    entropyCalibratorFiles = Taurus::cUtils::find_files(int8ImageDirectory, "*.jpg;*.png;*.bmp;*.jpeg;*.tiff");
                    if (entropyCalibratorFiles.empty())
                    {
                        INFOE("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory %s", int8ImageDirectory.c_str());
                        return false;
                    }
                    if (entropyCalibratorFiles.size() < maxBatchSize)
                    {
                        INFOW("Too few images provided, %d[provided] < %d[max batch size], image copy will be performed", entropyCalibratorFiles.size(), maxBatchSize);
                        int old_size = entropyCalibratorFiles.size();
                        for (int i = old_size; i < maxBatchSize; ++i)
                            entropyCalibratorFiles.push_back(entropyCalibratorFiles[i % old_size]);
                    }
                }
            }
            else
            {
                if (hasEntropyCalibrator)
                {
                    INFOW("int8EntropyCalibratorFile is ignore, when mode is '%s'", mode_string(mode));
                }
            }

            INFOI("Compile %s %s.", mode_string(mode), source.c_str());
            /******************************** builder ***********************************/
            auto builder = make_nvshared(nvinfer1::createInferBuilder(glogger));
            if (builder == nullptr)
            {
                INFOE("Can not create builder.");
                return false;
            }

            auto config = make_nvshared(builder->createBuilderConfig());
            if (mode == Mode::FP16)
            {
                if (!builder->platformHasFastFp16()) // jetson FP16
                    INFOW("Platform not have fast fp16 support");
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            else if (mode == Mode::INT8)
            {
                if (!builder->platformHasFastInt8())
                    INFOW("platform not have fast Int8 support");
                config->setFlag(nvinfer1::BuilderFlag::kINT8);
            }

            std::shared_ptr<nvinfer1::INetworkDefinition> network;
            const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            network = make_nvshared(builder->createNetworkV2(explicitBatch));
            // just support onnx
            std::shared_ptr<nvonnxparser::IParser> onnxParser = make_nvshared(nvonnxparser::createParser(*network, glogger));
            if (onnxParser == nullptr)
            {
                INFOE("Can not create parser.");
                return false;
            }

            if (!onnxParser->parseFromFile(source.c_str(), 1))
            {
                INFOE("Can not parse OnnX file: %s", source.c_str());
                return false;
            }

            auto inputTensor = network->getInput(0);
            auto inputDims = inputTensor->getDimensions();
            std::shared_ptr<Int8EntropyCalibrator> int8Calibrator;
            if (mode == Mode::INT8)
            {
                auto calibratorDims = inputDims;
                calibratorDims.d[0] = maxBatchSize;
                if (hasEntropyCalibrator)
                {
                    INFOI("Using exist entropy calibrator data[%d bytes]: %s", entropyCalibratorData.size(), int8EntoryCalibratorFile.c_str());
                    int8Calibrator.reset(new Int8EntropyCalibrator(entropyCalibratorData, calibratorDims, int8process));
                }
                else
                {
                    INFOI("Using image list[%d files]: %s", entropyCalibratorFiles.size(), int8ImageDirectory.c_str());
                    int8Calibrator.reset(new Int8EntropyCalibrator(entropyCalibratorFiles, calibratorDims, int8process, int8EntoryCalibratorFile));
                }
                config->setInt8Calibrator(int8Calibrator.get());
            }

            INFOI("Input shape is %s", join_dims(std::vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
            INFOI("Set max batch size = %d", maxBatchSize);
            INFOI("Set max workspace size = %.2f MB", maxWorkSpaceSize / 1024.0f / 1024.0f);
            INFOI("Base device: %s", Taurus::cudaTools::description());

            // display
            int net_num_input = network->getNbInputs();
            INFOI("Network has %d inputs:", net_num_input);
            std::vector<std::string> input_names(net_num_input);
            for (int i = 0; i < net_num_input; ++i)
            {
                auto tensor = network->getInput(i);
                auto dims = tensor->getDimensions();
                auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
                INFOI("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());

                input_names[i] = tensor->getName();
            }

            int net_num_output = network->getNbOutputs();
            INFOI("Network has %d outputs:", net_num_output);
            for (int i = 0; i < net_num_output; ++i)
            {
                auto tensor = network->getOutput(i);
                auto dims = tensor->getDimensions();
                auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
                INFOI("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
            }

            int net_num_layers = network->getNbLayers();
            INFOI("Network has %d layers:", net_num_layers);
            for (int i = 0; i < net_num_layers; ++i)
            {
                auto layer = network->getLayer(i);
                auto name = layer->getName();
                auto type_str = layer_type_name(layer);
                auto input0 = layer->getInput(0);
                if (input0 == nullptr)
                    continue;

                auto output0 = layer->getOutput(0);
                auto input_dims = input0->getDimensions();
                auto output_dims = output0->getDimensions();
                bool has_input = layer_has_input_tensor(layer);
                bool has_output = layer_has_output_tensor(layer);
                auto descript = layer_descript(layer);
                type_str = align_blank(type_str, 18);
                auto input_dims_str = align_blank(dims_str(input_dims), 18);
                auto output_dims_str = align_blank(dims_str(output_dims), 18);
                auto number_str = align_blank(format("%d.", i), 4);

                const char *token = "      ";
                if (has_input)
                    token = "  >>> ";
                else if (has_output)
                    token = "  *** ";

                INFOV("%s%s%s %s-> %s%s", token,
                      number_str.c_str(),
                      type_str.c_str(),
                      input_dims_str.c_str(),
                      output_dims_str.c_str(),
                      descript.c_str());
            }

            builder->setMaxBatchSize(maxBatchSize);
            config->setMaxWorkspaceSize(maxWorkSpaceSize);

            auto profile = builder->createOptimizationProfile();
            for (int i = 0; i < net_num_input; ++i)
            {
                auto input = network->getInput(i);
                auto input_dims = input->getDimensions();
                input_dims.d[0] = 1;
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
                input_dims.d[0] = maxBatchSize;
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
            }

            config->addOptimizationProfile(profile);
            INFOI("Building engine .... ");
            auto time_start = Taurus::cUtils::timestamp_now();
            auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
            if (engine == nullptr)
            {
                INFOE("engine is nullptr");
                return false;
            }
            INFOI("Build done %lld ms !", Taurus::cUtils::timestamp_now() - time_start);
            auto seridata = make_nvshared(engine->serialize());
            return save_file(saveto, seridata->data(), seridata->size());
        }

    }; // namespace TRT
}; // namespace Taurus