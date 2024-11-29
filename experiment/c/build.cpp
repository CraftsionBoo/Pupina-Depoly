#include "NvOnnxParser.h" //For onnxparser
#include "NvInfer.h"      // For tensorRT src
#include "common.h"       // For tensorRT sample common
#include "logger.h"       // For tensorRT sample loggers
#include "buffers.h"      // For tensorRT sample buffers

#include <iostream>
#include <memory>   // For unique_str
#include <cassert>  // c assert
#include <fstream>  // For save engine
// #include <unistd.h> // For chdir
#include <string>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "User : ./build [onnx_file_path]" << std::endl;
        return -1;
    }

    char *onnx_file_path = argv[1]; // onnx
    // char *main_path = argv[2];      // trt路径
    // chdir(main_path);

    // create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

    // create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    // onnxparser创建网络
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    auto paresd = parser->parseFromFile(onnx_file_path, static_cast<int>(sample::gLogger.getReportableSeverity()));

    // config network params
    auto input = network->getInput(0);
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 28, 28));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 28, 28));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 28, 28));

    // optimize netwrok
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->addOptimizationProfile(profile);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 28); // 设置最大工作空间
    // auto profilestream = samplesCommon::makeCudaStream(); // 创建流用于设置profile
    // config->setProfileStream(*profilestream);

    // create engine
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    std::ofstream engine_file("model.engine", std::ios::binary);
    assert(engine_file.is_open() && "Failed to open file for waiting");
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();
    std::cout << "Engine build successfully!" << std::endl;

    return 0;
}