#include "feature_extractor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace cbir {

namespace {
void normalizeL1(std::vector<float>& v) {
    const float sum = std::accumulate(v.begin(), v.end(), 0.0f);
    if (sum <= 1e-9f) {
        return;
    }
    for (float& x : v) {
        x /= sum;
    }
}
}  // namespace

FeatureExtractor::FeatureExtractor(int bins_per_channel) : bins_per_channel_(bins_per_channel) {
    if (bins_per_channel_ <= 1) {
        throw std::invalid_argument("bins_per_channel must be > 1");
    }
}

FeatureVector FeatureExtractor::extract(const cv::Mat& image_bgr) const {
    FeatureVector f;
    f.color = extractColorLabHistogram(image_bgr);
    f.shape = extractShapeFeatures(image_bgr);
    f.texture = extractTextureFeatures(image_bgr);
    return f;
}

std::vector<float> FeatureExtractor::extractColorLabHistogram(const cv::Mat& image_bgr) const {
    cv::Mat lab;
    cv::cvtColor(image_bgr, lab, cv::COLOR_BGR2Lab);

    const int bins = bins_per_channel_;
    const int total_bins = bins * bins * bins;
    std::vector<float> hist(total_bins, 0.0f);

    for (int r = 0; r < lab.rows; ++r) {
        const cv::Vec3b* row_ptr = lab.ptr<cv::Vec3b>(r);
        for (int c = 0; c < lab.cols; ++c) {
            const cv::Vec3b pix = row_ptr[c];
            int b0 = std::min((pix[0] * bins) / 256, bins - 1);
            int b1 = std::min((pix[1] * bins) / 256, bins - 1);
            int b2 = std::min((pix[2] * bins) / 256, bins - 1);
            int idx = (b0 * bins + b1) * bins + b2;
            hist[idx] += 1.0f;
        }
    }
    normalizeL1(hist);
    return hist;
}

std::vector<float> FeatureExtractor::extractShapeFeatures(const cv::Mat& image_bgr) const {
    cv::Mat gray;
    cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Moments m = cv::moments(mask, true);
    std::vector<float> hu(7, 0.0f);
    double hu_raw[7] = {};
    cv::HuMoments(m, hu_raw);
    for (int i = 0; i < 7; ++i) {
        hu[i] = static_cast<float>(hu_raw[i]);
    }

    double mu20 = m.mu20 / (m.m00 + 1e-9);
    double mu02 = m.mu02 / (m.m00 + 1e-9);
    double mu11 = m.mu11 / (m.m00 + 1e-9);
    double common = std::sqrt(4.0 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02));
    double lambda1 = 0.5 * (mu20 + mu02 + common);
    double lambda2 = 0.5 * (mu20 + mu02 - common);
    double ecc = 0.0;
    if (lambda1 > 1e-9) {
        ecc = std::sqrt(std::max(0.0, 1.0 - (lambda2 / lambda1)));
    }

    hu.push_back(static_cast<float>(ecc));
    return hu;
}

std::vector<float> FeatureExtractor::extractTextureFeatures(const cv::Mat& image_bgr) const {
    cv::Mat gray;
    cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Scalar lap_mean, lap_stddev;
    cv::meanStdDev(lap, lap_mean, lap_stddev);

    std::vector<float> t(3, 0.0f);
    t[0] = static_cast<float>(mean[0] / 255.0);
    t[1] = static_cast<float>(stddev[0] / 255.0);
    t[2] = static_cast<float>(lap_stddev[0] / 255.0);
    return t;
}

}  // namespace cbir

