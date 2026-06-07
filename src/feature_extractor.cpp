#include "feature_extractor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

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

// cv::Mat buildForegroundMaskGrabCut(const cv::Mat& image_bgr) {
//       // Fallback về Otsu nếu ảnh quá nhỏ
//       if (image_bgr.rows < 32 || image_bgr.cols < 32) {
//           return buildForegroundMask(image_bgr);  // giữ nguyên hàm cũ
//       }

//       const int margin_x = image_bgr.cols / 6;
//       const int margin_y = image_bgr.rows / 6;
//       cv::Rect rect(margin_x, margin_y,
//                     image_bgr.cols - 2 * margin_x,
//                     image_bgr.rows - 2 * margin_y);

//       cv::Mat bgdModel, fgdModel;
//       cv::Mat result(image_bgr.size(), CV_8UC1, cv::GC_BGD);
//       cv::grabCut(image_bgr, result, rect, bgdModel, fgdModel, 3, cv::GC_INIT_WITH_RECT);

//       cv::Mat mask;
//       cv::compare(result, cv::GC_PR_FGD, mask, cv::CMP_EQ);
//       cv::Mat fg_mask;
//       cv::compare(result, cv::GC_FGD, fg_mask, cv::CMP_EQ);
//       cv::bitwise_or(mask, fg_mask, mask);

//       // Morphological cleanup
//       cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
//                        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

//       // Fallback nếu GrabCut cho mask rỗng
//       if (cv::countNonZero(mask) < 500) {
//         return buildForegroundMask(image_bgr);
//       }
//     return mask;
// }

cv::Mat buildForegroundMask(const cv::Mat& image_bgr) {
    cv::Mat gray;
    cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
  
    cv::Mat global_mask;
    cv::threshold(gray, global_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat center_prior(gray.size(), CV_8U, cv::Scalar(0));
    const int cx = gray.cols / 2;
    const int cy = gray.rows / 2;
    const int rx = static_cast<int>(gray.cols * 0.35);
    const int ry = static_cast<int>(gray.rows * 0.35);
    cv::ellipse(center_prior, cv::Point(cx, cy), cv::Size(rx, ry), 0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED);

    cv::Mat centered_mask;
    cv::bitwise_and(global_mask, center_prior, centered_mask);
    if (cv::countNonZero(centered_mask) < 200) {
        centered_mask = global_mask;
    }

    cv::morphologyEx(centered_mask, centered_mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::morphologyEx(centered_mask, centered_mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    return centered_mask;
}

float gaussianCenterWeight(int r, int c, int rows, int cols) {
    const float cy = static_cast<float>(rows - 1) * 0.5f;
    const float cx = static_cast<float>(cols - 1) * 0.5f;
    const float dy = (static_cast<float>(r) - cy) / std::max(1.0f, static_cast<float>(rows));
    const float dx = (static_cast<float>(c) - cx) / std::max(1.0f, static_cast<float>(cols));
    const float sigma = 0.22f;
    return std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
}

double meanAt(const cv::Mat& integral, int r0, int c0, int r1, int c1) {
    const double area = static_cast<double>((r1 - r0) * (c1 - c0));
    if (area <= 0.0) {
        return 0.0;
    }
    const double sum =
        integral.at<double>(r1, c1) - integral.at<double>(r0, c1) - integral.at<double>(r1, c0) + integral.at<double>(r0, c0);
    return sum / area;
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
    const cv::Mat fg_mask = buildForegroundMask(image_bgr);

    const int bins = bins_per_channel_;
    const int total_bins = bins * bins * bins;
    std::vector<float> hist(total_bins, 0.0f);

    for (int r = 0; r < lab.rows; ++r) {
        const cv::Vec3b* row_ptr = lab.ptr<cv::Vec3b>(r);
        const uint8_t* mask_ptr = fg_mask.ptr<uint8_t>(r);
        for (int c = 0; c < lab.cols; ++c) {
            // PWH: prefer center pixels while still retaining weak background context.
            float weight = 0.25f + gaussianCenterWeight(r, c, lab.rows, lab.cols);
            if (mask_ptr[c] > 0) {
                weight *= 1.8f;
            }

            const cv::Vec3b pix = row_ptr[c];
            int b0 = std::min((pix[0] * bins) / 256, bins - 1);
            int b1 = std::min((pix[1] * bins) / 256, bins - 1);
            int b2 = std::min((pix[2] * bins) / 256, bins - 1);
            int idx = (b0 * bins + b1) * bins + b2;
            hist[idx] += weight;
        }
    }
    normalizeL1(hist);
    return hist;
}

std::vector<float> FeatureExtractor::extractShapeFeatures(const cv::Mat& image_bgr) const {
    cv::Mat mask = buildForegroundMask(image_bgr);
    if (cv::countNonZero(mask) < 64) {
        return std::vector<float>(72, 0.0f);
    }

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

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        hu.resize(72, 0.0f);
        return hu;
    }

    const auto largest_it = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });
    const cv::Rect bbox = cv::boundingRect(*largest_it);
    cv::Mat roi = mask(bbox);

    const int grid_rows = 8;
    const int grid_cols = 8;
    for (int gr = 0; gr < grid_rows; ++gr) {
        for (int gc = 0; gc < grid_cols; ++gc) {
            const int r0 = gr * roi.rows / grid_rows;
            const int r1 = (gr + 1) * roi.rows / grid_rows;
            const int c0 = gc * roi.cols / grid_cols;
            const int c1 = (gc + 1) * roi.cols / grid_cols;
            const cv::Rect cell(c0, r0, std::max(1, c1 - c0), std::max(1, r1 - r0));
            const float density = static_cast<float>(cv::countNonZero(roi(cell))) /
                                  static_cast<float>(std::max(1, cell.width * cell.height));
            hu.push_back(density >= 0.15f ? 1.0f : 0.0f);
        }
    }
    return hu;
}

std::vector<float> FeatureExtractor::extractTextureFeatures(const cv::Mat& image_bgr) const {
    cv::Mat gray;
    cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);
    const cv::Mat fg_mask = buildForegroundMask(image_bgr);

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray_f, mean, stddev, fg_mask);
    const float contrast = static_cast<float>(stddev[0]);

    cv::Mat integral_img;
    cv::integral(gray_f, integral_img, CV_64F);
    std::vector<int> scales = {1, 2, 4, 8};
    const int max_scale = scales.back();
    const int margin = 2 * max_scale;
    std::vector<double> energy_sum(scales.size(), 0.0);
    std::vector<int> energy_count(scales.size(), 0);

    for (int r = margin; r < gray.rows - margin; ++r) {
        const uint8_t* mptr = fg_mask.ptr<uint8_t>(r);
        for (int c = margin; c < gray.cols - margin; ++c) {
            if (mptr[c] == 0) {
                continue;
            }
            for (size_t s = 0; s < scales.size(); ++s) {
                const int k = scales[s];
                const double a = meanAt(integral_img, r - k, c - k, r + k + 1, c + k + 1);
                const double h1 = meanAt(integral_img, r - k, c - 2 * k, r + k + 1, c);
                const double h2 = meanAt(integral_img, r - k, c + 1, r + k + 1, c + 2 * k + 1);
                const double v1 = meanAt(integral_img, r - 2 * k, c - k, r, c + k + 1);
                const double v2 = meanAt(integral_img, r + 1, c - k, r + 2 * k + 1, c + k + 1);
                const double e = std::max({std::abs(a - h1), std::abs(a - h2), std::abs(a - v1), std::abs(a - v2)});
                energy_sum[s] += e;
                energy_count[s] += 1;
            }
        }
    }

    float coarseness = 0.0f;
    if (std::accumulate(energy_count.begin(), energy_count.end(), 0) > 0) {
        double weighted_scale = 0.0;
        double total_energy = 0.0;
        for (size_t s = 0; s < scales.size(); ++s) {
            if (energy_count[s] == 0) {
                continue;
            }
            const double avg_e = energy_sum[s] / static_cast<double>(energy_count[s]);
            weighted_scale += avg_e * static_cast<double>(scales[s]);
            total_energy += avg_e;
        }
        if (total_energy > 1e-9) {
            coarseness = static_cast<float>(weighted_scale / total_energy / static_cast<double>(scales.back()));
        }
    }

    std::vector<float> t(2, 0.0f);
    t[0] = std::clamp(coarseness, 0.0f, 1.0f);
    t[1] = std::clamp(contrast, 0.0f, 1.0f);
    return t;
}

}  // namespace cbir

