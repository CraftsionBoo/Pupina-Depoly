#include "NvInfer.h"      // For tensorRT src
#include "NvOnnxParser.h" // For onnxparser
#include "logger.h"       // For tensorRT sample loggers
#include "common.h"       // For tensorRT sample common
#include "buffers.h"      // For tensorRT sample buffers

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unistd.h> // for assert
#include <memory>   // for unique_ptr

// load model.onnx
std::vector<unsigned char> loadEngineModel(const std::string &filename)
{
    std::ifstream load_file(filename, std::ios::binary);
    assert(load_file.is_open() && "load engine model failed");
    load_file.seekg(0, std::ios::end); // 定位文件末尾
    size_t size = load_file.tellg();   // 文件大小

    std::vector<unsigned char> data(size);
    load_file.seekg(0, std::ios::beg); // 定位文件开头
    load_file.read((char *)data.data(), size);
    load_file.close();
    return data;
}

// load mnist pgm
void mreadPGMFile(const std::string &filename, uint8_t *buffer, int inH, int inW)
{
    std::ifstream infile(filename, std::ios::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char *>(buffer), inH * inW);
}


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "User : " << argv[0] << " <engine_path> <image_path>" << std::endl;
        return 1;
    }

    char *engine_file = argv[1]; // 模型
    char *image_path = argv[2];  // 图像

    // runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

    // deserialize
    std::vector<unsigned char> plan = loadEngineModel(engine_file);
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));

    // create context
    auto mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    // create buffer
    samplesCommon::BufferManager buffers(mEngine);

    // runtime
    nvinfer1::Dims mInputDims;
    mInputDims.nbDims = 4; // 设置维度数量为 4
    mInputDims.d[0] = 1;   // 第一个维度为 1
    mInputDims.d[1] = 1;   // 第二个维度为 1
    mInputDims.d[2] = 28;  // 第三个维度为 28
    mInputDims.d[3] = 28;  // 第四个维度为 28

    nvinfer1::Dims mOutputDims;
    mOutputDims.nbDims = 2; // 设置维度数量为 2
    mOutputDims.d[0] = 1;   // 第一个维度为 1
    mOutputDims.d[1] = 10;  // 第二个维度为 10

    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::vector<uint8_t> fileData(inputH * inputW);
    mreadPGMFile(image_path, fileData.data(), inputH, inputW); // 读取数据
    // display
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    sample::gLogInfo << std::endl;

    // name vector
    std::vector<std::string> inputTensorName;
    std::vector<std::string> outputTensorName;

    // 模型转为onnx可以使用netron进行查看 输入输出name
    inputTensorName.push_back("input");
    outputTensorName.push_back("output");

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(inputTensorName[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    // infer 
    buffers.copyInputToDevice();
    bool status = mContext->executeV2(buffers.getDeviceBindings().data());
    if (!status)
        return -1;
    buffers.copyOutputToHost();

    // output name
    const int outputsize = mOutputDims.d[1];
    float *output = static_cast<float *>(buffers.getHostBuffer(outputTensorName[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum(0.0f);
    for (int i = 0; i < outputsize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }
    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputsize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
            idx = i;
        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return 0;
}