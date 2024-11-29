#include "app_yolo_seg/yolo_seg.hpp"

#include "../tensorRT/common/ilogger.hpp"
#include "../tensorRT/common/cv_cpp_utils.hpp"
#include "../tensorRT/infer/trt_infer.hpp"
#include "../tensorRT/builder/trt_builder.hpp"

using namespace Taurus;

static const char *fish_name[] = {
    "Surgeonfishes_allied",
    "Parrotfishes_allied",
    "Gnathodentex aureolineatus Striped Large-eye Bream",
    "Damselfishes_allied",
    "Chaetodon trifascialis Chevron Butterflyfish",
    "Acanthurus leucosternon Powder Blue Tang",
    "ButterflyFishes",
    "Stegastes punctatus Bluntsnout Gregory",
    "Ctenochaetus striatus Lined Bristletooth",
    "Chaetodon meyeri Meyer-s Butterflyfish",
    "Melichthys indicus Black-finned Triggerfish",
    "Chaetodon collare Red-tailed Butterflyfish",
    "Myripristis violacea Violet Soldierfish",
    "Lutjanus gibbus Humpback Snapper",
    "Snappers_allied",
    "Triggerfishes_allied",
    "Squirrel_soldierfishes",
    "Chlorurus sordidus Indian Bullethead Parrotfish",
    "Scarus rubroviolaceus Redlip Parrotfish",
    "Myripristis murdjan Crimson Soldierfish",
    "Sargocentron caudimaculatum Tailspot Squirrelfish",
    "Scarus frenatus Sixband Parrotfish",
    "Scarus tricolor Tricolour Parrotfish",
    "Abudefduf vaigiensis Indo-Pacific Sergeant Major",
    "Lutjanus kasmira Bluestriped Snapper",
    "Hipposcarus harid Indian Longnose Parrotfish",
    "Goatfishes_allied",
    "Scarus Parrotfishes_allied JUV",
    "Mulloidichthys vanicolensis Yellowfin Goatfish",
    "Groupers_allied",
    "Neoniphon sammara Spotfin Squirrelfish",
    "Dascyllus abudafur Indian Ocean Humbug",
    "Epinephelus merra Honeycomb Grouper",
    "Gomphosus caeruleus Green Bird Wrasse",
    "Thalassoma hardwicke Sixbar Wrasse",
    "Acanthurus lineatus Striped Surgeonfish",
    "Chaetodon auriga Threadfin Butterflyfish",
    "Chaetodon xanthocephalus Yellowhead Butterflyfish",
    "Chaetodon lunula Raccoon Butterflyfish",
    "Scarus scaber Five-saddle Parrotfish",
    "Halichoeres hortulanus Checkerboard Wrasse",
    "Acanthurus triostegus Convict Surgeonfish",
    "Chlorurus capistratoides Pink-margined Parrotfish",
    "Parupeneus macronemus Longbarbel Goatfish",
    "Otherfishes",
    "Amphiprion nigripes Blackfin Anemonefish",
    "Eel",
    "Balistapus undulatus Orangestripe Triggerfish",
    "Plectorhinchus vittatus Oriental Sweetlips",
    "Sweetlips",
    "Chromis ternatensis Ternate Chromis",
    "Chaetodon falcula Saddleback Butterflyfish",
    "Naso elegans Elegant Unicornfish",
    "Lutjanus bohar Two-spot Red Snapper",
    "Naso brevirostris Paletail Unicornfish",
    "Pomacanthus imperator Emperor Angelfish",
    "Halichoeres lamarii Dark Green Wrasse",
    "Zanclus cornutus Moorish Idol",
    "Cirrhitichthys falco Dwarf Hawkfish",
    "Dascyllus carneus Indian Dascyllus",
    "Chromis viridis Blue-green Chromis",
    "Spratelloides gracilis Slender Sprat",
    "Arothron nigropunctatus Blackspotted Puffer",
    "Acanthurus nigricauda Eyeline Surgeonfish India",
    "Acanthurus blochii Dark Surgeonfish",
    "Labroides bicolor Bicolor Cleaner Wrasse",
    "Malacanthus latovittatus Blue Blanquillo",
    "Squids",
    "Parupeneus trifasciatus Indian Doublebar Goatfish",
    "Chromis fieldi Indian half-and-half chromis",
    "Chaetodon trifasciatus Indian Redfin Butterflyfish",
    "Lutjanus monostigma Onespot Snapper",
    "butterflyfish_allied JUV",
    "Acanthurus tennentii Lieutenant Surgeonfish",
    "Diodon hystrix Spotted Porcupinefish",
    "Abudefduf sordidus Blackspot Sergeant",
    "Abudefduf septemfasciatus Nine-band Sergeant",
    "Monotaxis heterodon Redfin Bream",
    "Sargocentron spiniferum Longjaw squirrelfish",
    "DCA",
    "water_background",
    "CE"};

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

static void draw_mask(cv::Mat &image, YoloSeg::Box &obj, cv::Scalar &color)
{
    // compute IM
    float scale_x = 640 / (float)image.cols;
    float scale_y = 640 / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float ox = -scale * image.cols * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    float oy = -scale * image.rows * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    cv::Mat M = (cv::Mat_<float>(2, 3) << scale, 0, ox, 0, scale, oy);

    cv::Mat IM;
    cv::invertAffineTransform(M, IM);

    cv::Mat mask_map = cv::Mat::zeros(cv::Size(160, 160), CV_8UC1);
    cv::Mat small_mask(obj.seg->height, obj.seg->width, CV_8UC1, obj.seg->data);
    cv::Rect roi(obj.seg->left, obj.seg->top, obj.seg->width, obj.seg->height);
    small_mask.copyTo(mask_map(roi));
    cv::resize(mask_map, mask_map, cv::Size(640, 640)); // 640x640
    cv::threshold(mask_map, mask_map, 128, 1, cv::THRESH_BINARY);

    cv::Mat mask_resized;
    cv::warpAffine(mask_map, mask_resized, IM, image.size(), cv::INTER_LINEAR);

    // create color mask
    cv::Mat colored_mask = cv::Mat::ones(image.size(), CV_8UC3);
    colored_mask.setTo(color);

    cv::Mat masked_colored_mask;
    cv::bitwise_and(colored_mask, colored_mask, masked_colored_mask, mask_resized);

    // create mask indices
    cv::Mat mask_indices;
    cv::compare(mask_resized, 1, mask_indices, cv::CMP_EQ);

    cv::Mat image_masked, colored_mask_masked;
    image.copyTo(image_masked, mask_indices);
    masked_colored_mask.copyTo(colored_mask_masked, mask_indices);

    // weighted sum
    cv::Mat result_masked;
    cv::addWeighted(image_masked, 0.6, colored_mask_masked, 0.4, 0, result_masked);

    // copy result to image
    result_masked.copyTo(image, mask_indices);
}

static void inference_and_performance(int deviceid, const std::string &engine_file, TRT::Mode mode,
                                      const std::string &model_name, const std::string &samples)
{
    auto engine = YoloSeg::create_seg_infer(
        engine_file,                 // engine file
        deviceid,                    // gpu id
        0.25f,                       // confidence threshold
        0.45f,                       // nms threshold
        YoloSeg::NMSMethod::FastGPU, // NMS method cpu / gpu
        1024,                        // max objects
        false                        // preprocess use multi stream
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
        std::vector<std::shared_future<YoloSeg::Boxseg>> boxes_array;
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
        const char *type_name = "yolo_seg";
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
                cv::Scalar color(b, g, r);
                if (obj.seg)
                {
                    draw_mask(image, obj, color);
                }
            }

            for (auto &obj : boxes)
            {
                uint8_t b, g, r;
                std::tie(b, g, r) = cUtils::random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name = fish_name[obj.class_label];
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
    else
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
                cv::Scalar color(b, g, r);
                if (obj.seg)
                {
                    draw_mask(frame, obj, color);
                }
            }

            for (auto &obj : boxes)
            {
                uint8_t b, g, r;
                std::tie(b, g, r) = cUtils::random_color(obj.class_label);
                cv::rectangle(frame, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name = fish_name[obj.class_label];
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

static void test(TRT::Mode mode, const std::string &model)
{
    std::string workspace_path = "workspace/fish_sea/models"; // 修改
    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const std::vector<std::string> &files, std::shared_ptr<TRT::Tensor> &tensor)
    {
        INFOI("INT8 %d / %d", current, count);
        for (int i = 0; i < files.size(); ++i)
        {
            auto image = cv::imread(files[i]);
            YoloSeg::image_to_tensor(image, tensor, i);
        }
    };

    const char *name = model.c_str();
    INFOI("===================== test Yolo-seg %s %s ==================================", mode_name, name);
    std::string onnx_file = cUtils::format2048("%s/%s.onnx", workspace_path.c_str(), name);
    std::string model_file = cUtils::format2048("%s/%s_%s.trtmodel", workspace_path.c_str(), name, mode_name);
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
    std::string sample_files = "workspace/fish_sea/assets"; // 修改

    inference_and_performance(deviceid, model_file, mode, name, sample_files);
}

int app_yolo_seg()
{
    test(TRT::Mode::FP16, "yolov8-seg");
}