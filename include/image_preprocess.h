#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace cbir {

class ImagePreprocessor {
   public:
    ImagePreprocessor(int target_width, int target_height);

    cv::Mat loadAndPreprocess(const std::string& image_path) const;

   private:
    int target_width_;
    int target_height_;
};

}  // namespace cbir

