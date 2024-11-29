#include "app_depthanything/depth_v2.hpp"

#include "../tensorRT/common/ilogger.hpp"
#include "../tensorRT/common/cv_cpp_utils.hpp"
#include "../tensorRT/infer/trt_infer.hpp"
#include "../tensorRT/builder/trt_builder.hpp"

using namespace Taurus;

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

static std::tuple<cv::Mat, int, int> resize_depth(cv::Mat &img, int w, int h)
{
    cv::Mat result;
    int nw, nh;
    int ih = img.rows;
    int iw = img.cols;
    float aspectRatio = (float)img.cols / (float)img.rows;

    if (aspectRatio >= 1)
    {
        nw = w;
        nh = int(h / aspectRatio);
    }
    else
    {
        nw = int(w * aspectRatio);
        nh = h;
    }
    cv::resize(img, img, cv::Size(nw, nh));
    result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
    cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(h, w, CV_8UC3, 0.0);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

    std::tuple<cv::Mat, int, int> res_tuple = std::make_tuple(out, (w - nw) / 2, (h - nh) / 2);

    return res_tuple;
}

static void inference_and_performance(int deviceid, const std::string &engine_file, TRT::Mode mode,
                                      const std::string &model_name, const std::string &samples)
{
    static int input_w = 518, input_h = 518;

    auto engine = depth::create_depth_infer(
        engine_file, // engine file
        deviceid,    // gpu id
        false        // preprocess use multi stream
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
        std::vector<std::shared_future<cv::Mat>> depth_array;
        for (int i = 0; i < 2; ++i)
            depth_array = engine->commits(images);
        depth_array.back().get();
        depth_array.clear();

        // inference
        const int ntest = 1;
        auto begin_timer = cUtils::timestamp_now_float();
        for (int i = 0; i < ntest; ++i)
            depth_array = engine->commits(images);

        // wait all result
        depth_array.back().get();
        float inference_average_time = (cUtils::timestamp_now_float() - begin_timer) / ntest / images.size();
        const char *type_name = "depth_v2";
        auto mode_name = TRT::mode_string(mode);
        INFOI("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name, inference_average_time, 1000 / inference_average_time);
        append_to_file("perf.result.log", cUtils::format2048("%s,%s,%s,%f", model_name.c_str(), type_name, mode_name, inference_average_time));

        std::string root = cUtils::format2048("workspace/%s_%s_%s_result", model_name.c_str(), type_name, mode_name);
        cUtils::rmtree(root);
        cUtils::mkdir(root);

        for (int i = 0; i < depth_array.size(); ++i)
        {
            auto &image = images[i];

            int img_w = image.cols;
            int img_h = image.rows;

            auto depth = depth_array[i].get();

            cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::Mat colormap;
            cv::applyColorMap(depth, colormap, cv::COLORMAP_INFERNO);

            // Rescale the colormap
            // int limX, limY;
            // if (img_w > img_h)
            // {
            //     limX = input_w;
            //     limY = input_w * img_h / img_w;
            // }
            // else
            // {
            //     limX = input_w * img_w / img_h;
            //     limY = input_w;
            // }
            // cv::resize(colormap, colormap, cv::Size());

            std::string file_name = cUtils::file_name(files[i], false);
            std::string save_path = cUtils::format2048("%s/%s.jpg", root.c_str(), file_name.c_str());
            INFOI("Save to %s, average time %.2f ms", save_path.c_str(), inference_average_time);
            cv::imwrite(save_path, colormap);
        }
        engine.reset();
    }
    else
    {
        cv::Mat frame;
        cv::VideoCapture cap(samples);

        // 视频保存
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = 24;
        cv::VideoWriter writer("output.avi", fourcc, fps, cv::Size(518, 518));

        if (!cap.isOpened())
        {
            INFOE("Camera open failed");
            return;
        }

        while (cap.read(frame))
        {
            auto t0 = cUtils::timestamp_now_float();
            auto depth_single = engine->commit(frame).get();

            cv::normalize(depth_single, depth_single, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::Mat colormap;
            cv::applyColorMap(depth_single, colormap, cv::COLORMAP_INFERNO);

            auto fee = cUtils::timestamp_now_float() - t0;
            INFOI("fee = %.2f ms, FPS = %.2f", fee, 1 / fee * 1000);
            // cv::imshow("frame", colormap);
            writer.write(colormap);
            int key = cv::waitKey(1);
            if (key == 27)
                break;
        }
        INFOI("video Done");
        cap.release();
        // cv::destroyAllWindows();
        writer.release();
        engine.reset();
    }
}

static void test(TRT::Mode mode, const std::string &model)
{
    std::string workspace_path = "workspace/depth_v2/models"; // 修改
    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const std::vector<std::string> &files, std::shared_ptr<TRT::Tensor> &tensor)
    {
        INFOI("INT8 %d / %d", current, count);
        for (int i = 0; i < files.size(); ++i)
        {
            auto image = cv::imread(files[i]);
            depth::image_to_tensor(image, tensor, i);
        }
    };

    const char *name = model.c_str();
    INFOI("===================== test depth %s %s ==================================", mode_name, name);
    std::string onnx_file = cUtils::format2048("%s/%s.onnx", workspace_path.c_str(), name);
    std::string model_file = cUtils::format2048("%s/%s_%s.trtmodel", workspace_path.c_str(), name, mode_name);
    int batch_size = 1;

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
    std::string sample_files = "workspace/depth_v2/assets"; // 修改
    // std::string sample_files = "workspace/depth_v2/davis_seasnake.mp4";

    inference_and_performance(deviceid, model_file, mode, name, sample_files);
}

int app_depthanything_v2()
{
    // test(TRT::Mode::FP16, "depth_anything_vits14");
    test(TRT::Mode::FP16, "depth_anything_v2_vits");
    // test(TRT::Mode::FP16, "depth_anything_v2_vitb");
    // test(TRT::Mode::FP16, "depth_anything_v2_vitl"); // 权重过大，显存不足
}