#pragma once

#include <opencv2/core.hpp>

#include "types.h"

namespace cbir {

class FeatureExtractor {
   public:
    explicit FeatureExtractor(int bins_per_channel = 8);

    FeatureVector extract(const cv::Mat& image_bgr) const;

   private:
    int bins_per_channel_;

    std::vector<float> extractColorLabHistogram(const cv::Mat& image_bgr) const;
    std::vector<float> extractShapeFeatures(const cv::Mat& image_bgr) const;
    std::vector<float> extractTextureFeatures(const cv::Mat& image_bgr) const;
};

}  // namespace cbir

