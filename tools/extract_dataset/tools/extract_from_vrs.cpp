#include "data_provider/VrsDataProvider.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace projectaria::tools;
using json = nlohmann::json;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./extract_from_vrs input.vrs output_dir\n";
        return 1;
    }

    std::string vrsPath = argv[1];
    std::string outputDir = argv[2];
    std::string imageDir = outputDir + "/images";
    std::string annotationFile = outputDir + "/object_location.jsonl";

    fs::create_directories(imageDir);
    std::ofstream out(annotationFile);

    auto provider = vrs::createVrsDataProvider(vrsPath);
    if (!provider) {
        std::cerr << "Failed to open VRS file.\n";
        return 1;
    }

    auto rgbStreamId = *provider->getStreamIdFromLabel("camera-rgb");
    auto records = provider->getSensorDataByStream(rgbStreamId);

    int64_t lastTime = 0;
    int64_t intervalNs = 1e9; // 1 second
    int frameIndex = 0;
    std::string sessionId = fs::path(vrsPath).stem().string();

    for (const auto& data : records) {
        int64_t ts = data.getTimeNs();
        if (ts - lastTime < intervalNs) continue;

        auto image = data.image().getImageMatrix();
        std::string filename = sessionId + "_frame_" + std::to_string(frameIndex) + ".jpg";
        std::string fullPath = imageDir + "/" + filename;
        cv::imwrite(fullPath, image);

        json annotation = {
            {"id", sessionId + "_F" + std::to_string(frameIndex)},
            {"video_id", sessionId},
            {"frame_index", frameIndex},
            {"scene_image", fullPath},
            {"user_query", "Where is my phone?"},
            {"descriptive_ground_truth", ""},
            {"action_ground_truth", ""},
            {"task_type", "object_localization"},
            {"environment", "indoor"},
            {"lighting", "unknown"},
            {"measurable_result", "Object location accuracy"},
            {"future_fields", {
                {"timestamp", ts / 1e9},
                {"video_path", vrsPath}
            }}
        };

        out << annotation.dump() << "\n";

        lastTime = ts;
        frameIndex++;
    }

    std::cout << "Extracted " << frameIndex << " frames to " << imageDir << "\n";
    return 0;
}
