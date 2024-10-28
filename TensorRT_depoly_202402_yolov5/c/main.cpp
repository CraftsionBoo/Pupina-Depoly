#include "v5.hpp"

#include "tensorRT/common/ilogger.hpp"
#include "tensorRT/common/cv_cpp_utils.hpp"
#include "tensorRT/infer/trt_infer.hpp"
#include "tensorRT/builder/trt_builder.hpp"

using namespace Taurus;

static const char *water_garbage[] = {"bottle", "can"};

static void append_to_file(const std::string &file, const std::string &data)
{
    FILE *f = fopen(file.c_str(), "a+");
    if (f == nullptr)
    {
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

bool build_model()
{
    if (cUtils::exists("workspace/models/water_garbage/water_garbage_self.trtmodel"))
    {
        INFOI("water_garbage.trtmodel has exists.\n");
        return true;
    }

    if (TRT::compile(TRT::Mode::FP16, 10, "workspace/models/water_garbage/water_garbage_self.onnx", 
    "workspace/models/water_garbage/water_garbage_self.trtmodel", 1 << 28))
    {
        INFOI("compile onnx done.");
        return true;
    }
    else
    {
        INFOE("compile failed");
        return false;
    }
}

static void inference(int deviceid, const std::string &engine_file, TRT::Mode mode, const std::string &model_name)
{
    auto engine = v5::create_infer(engine_file, deviceid, 0.25f, 0.45f, v5::NMSMethod::FastGPU, 1024, false);
    if(engine == nullptr)
    {
        INFOE("Engine is nullptr");
        return;
    }

    INFOI("load engine finish");
    auto files = cUtils::find_files("workspace/assets/water_garbage", "*.jpg;*.jpeg;*.png;*.gif;*.tif"); // 推理图像
    INFOI("infer images size : %d", files.size());
    std::vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i)
    {
        auto image = cv::imread(files[i]);
        images.push_back(image);
    }

    // warmup
    std::vector<std::shared_future<v5::BoxArray>> boxes_array;
    for (int i = 0; i < 2; ++i)
        boxes_array = engine->commits(images);
    boxes_array.back().get();
    boxes_array.clear();

    const int ntest = 1;
    auto begin_timer = cUtils::timestamp_now_float();
    for (int i = 0; i < ntest; ++i)
        boxes_array = engine->commits(images);

    // wait all result
    boxes_array.back().get();
    float inference_average_time = (cUtils::timestamp_now_float() - begin_timer) / ntest / images.size();
    std::string type_name = "yolov5";
    auto mode_name = TRT::mode_string(mode);
    INFOI("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name.c_str(), inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", cUtils::format2048("%s,%s,%s,%f", model_name.c_str(), type_name.c_str(), mode_name, inference_average_time));

    std::string root = cUtils::format2048("workspace/%s_%s_%s_result", model_name.c_str(), type_name.c_str(), mode_name);
    cUtils::rmtree(root);
    cUtils::mkdir(root);

    for(int i = 0; i < boxes_array.size(); ++i)
    {
        auto &image = images[i];
        auto boxes = boxes_array[i].get();

        for(auto& obj : boxes)
        {
            uint8_t b,g,r;
            std::tie(b, g, r) = cUtils::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto name = water_garbage[obj.class_label];
            auto caption = cUtils::format2048("%s. %.2f", name, obj.confidence);
            int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left - 3, obj.top -33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        std::string file_name = cUtils::file_name(files[i], false);
        std::string save_path = cUtils::format2048("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFOI("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
        cv::imwrite(save_path, image);
    }
    engine.reset();
}

int main(int argc, char** argv)
{
    if(build_model())
    {
        int deviceid = 0;
        TRT::Mode mode = TRT::Mode::FP16;

        inference(deviceid, "workspace/models/water_garbage/water_garbage_self.trtmodel", mode, "water_garbage");
    }
    return 0;
}