#include "app_yolo_det/yolo.hpp"

#include "../tensorRT/common/ilogger.hpp"
#include "../tensorRT/common/cv_cpp_utils.hpp"
#include "../tensorRT/infer/trt_infer.hpp"
#include "../tensorRT/builder/trt_builder.hpp"

using namespace Taurus;

static const char *water_garbage[] = {"bottle", "can"};

static const char *cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

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

static void inference_and_performance(int deviceid, const std::string &engine_file, TRT::Mode mode, Yolo::Type type,
                                      const std::string &model_name, const std::string &samples)
{
    auto engine = Yolo::create_infer(
        engine_file,              // engine file
        type,                     // yolo type, Yolo::Type::v5, Yolo::Type::v11
        deviceid,                 // gpu id
        0.25f,                    // confidence threshold
        0.45f,                    // nms threshold
        Yolo::NMSMethod::FastGPU, // NMS method, fast GPU / CPU
        1024,                     // max objects
        false                     // preprocess use multi stream
    );

    if (engine == nullptr)
    {
        INFOE("Engine is nullptr");
        return;
    }

    if (cUtils::isDirectory(samples)) // samples文件夹还是视频
    {
        auto files = cUtils::find_files(samples, "*.jpg;*.jpeg;*.png;*.gif;");
        INFOI("This folder contains images of %d", files.size());
        std::vector<cv::Mat> images;
        for (int i = 0; i < files.size(); ++i)
        {
            auto image = cv::imread(files[i]);
            images.push_back(image);
        }

        // warm up
        std::vector<std::shared_future<Yolo::BoxArray>> boxes_array;
        for (int i = 0; i < 2; ++i)
            boxes_array = engine->commits(images);
        boxes_array.back().get();
        boxes_array.clear();

        // inference
        const int ntest = 1;
        auto begin_timer = cUtils::timestamp_now_float();
        for (int i = 0; i < ntest; ++i)
            boxes_array = engine->commits(images);

        // wait all result
        boxes_array.back().get();
        float inference_average_time = (cUtils::timestamp_now_float() - begin_timer) / ntest / images.size();
        auto type_name = Yolo::type_name(type);
        auto mode_name = TRT::mode_string(mode);
        INFOI("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name, inference_average_time, 1000 / inference_average_time);
        append_to_file("perf.result.log", cUtils::format2048("%s,%s,%s,%f", model_name.c_str(), type_name, mode_name, inference_average_time));

        std::string root = cUtils::format2048("workspace/%s_%s_%s_result", model_name.c_str(), type_name, mode_name);
        cUtils::rmtree(root);
        cUtils::mkdir(root);

        for (int i = 0; i < boxes_array.size(); ++i)
        {
            auto &image = images[i];
            auto boxes = boxes_array[i].get();

            for (auto &obj : boxes)
            {
                uint8_t b, g, r;
                std::tie(b, g, r) = cUtils::random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name = water_garbage[obj.class_label];
                auto caption = cUtils::format2048("%s. %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }

            std::string file_name = cUtils::file_name(files[i], false);
            std::string save_path = cUtils::format2048("%s/%s.jpg", root.c_str(), file_name.c_str());
            INFOI("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
            cv::imwrite(save_path, image);
        }
        engine.reset();
    }
    else // 视频
    {
        cv::Mat frame;
        cv::VideoCapture cap(samples);
        if (!cap.isOpened())
        {
            INFOE("Camera open failed");
            return;
        }

        while (cap.read(frame))
        {
            auto t0 = cUtils::timestamp_now_float();
            auto boxes = engine->commit(frame).get();
            for (auto &obj : boxes)
            {
                uint8_t b, g, r;
                std::tie(b, g, r) = cUtils::random_color(obj.class_label);
                cv::rectangle(frame, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name = water_garbage[obj.class_label];
                auto caption = cUtils::format2048("%s. %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(frame, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(frame, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
            auto fee = cUtils::timestamp_now_float() - t0;
            INFOI("fee = %.2f ms, FPS = %.2f", fee, 1 / fee * 1000);
            cv::imshow("frame", frame);
            int key = cv::waitKey(1);
            if (key == 27)
                break;
        }
        INFOI("video Done");
        cap.release();
        cv::destroyAllWindows();
        engine.reset();
    }
}

static void test_app_yolo(Yolo::Type type, TRT::Mode mode, const std::string &model)
{
    // -----------------------> 测试路径修改
    std::string workspace_path = "workspace/water_garbage/models"; // 修改
    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const std::vector<std::string> &files, std::shared_ptr<TRT::Tensor> &tensor)
    {
        INFOI("INT8 %d / %d", current, count);
        for (int i = 0; i < files.size(); ++i)
        {
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, i);
        }
    };

    const char *name = model.c_str();
    INFOI("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name, name);

    std::string onnx_file = cUtils::format2048("%s/%s.onnx", workspace_path.c_str(), name);                    // yolov5.onnx
    std::string model_file = cUtils::format2048("%s/%s_%s.trtmodel", workspace_path.c_str(), name, mode_name); // yolov5_fp16.trtmodel
    int batch_size = 8;

    if (not cUtils::exists(model_file))
    {
        TRT::compile(
            mode,
            batch_size,
            onnx_file,
            model_file,
            1 << 28,
            int8process,
            "inference",
            "inference/int8.cache");
        INFOI("compile onnx done");
    }
    // -------------------------> 测试路径修改
    std::string sample_files = "workspace/water_garbage/assets"; // 修改

    inference_and_performance(deviceid, model_file, mode, type, name, sample_files);
}

int app_yolo_det()
{
    //water_garbage yolov5
    test_app_yolo(Yolo::Type::V5, TRT::Mode::FP16,"yolov5s");
}