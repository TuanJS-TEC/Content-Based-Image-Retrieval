#include "image_preprocess.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace cbir {

ImagePreprocessor::ImagePreprocessor(int target_width, int target_height)
    : target_width_(target_width), target_height_(target_height) {}

cv::Mat ImagePreprocessor::loadAndPreprocess(const std::string& image_path) const {
    cv::Mat raw = cv::imread(image_path, cv::IMREAD_COLOR);
    if (raw.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path);
    }

    cv::Mat resized;
    cv::resize(raw, resized, cv::Size(target_width_, target_height_), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

}  // namespace cbir

