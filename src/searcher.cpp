#include "searcher.h"

#include <algorithm>
#include <cmath>

namespace cbir {

Searcher::Searcher(const ImagePreprocessor& preprocessor, const FeatureExtractor& extractor, SqliteRepo& repo, const Config& cfg)
    : preprocessor_(preprocessor), extractor_(extractor), repo_(repo), cfg_(cfg) {}

bool Searcher::runQuery(const std::string& query_image_path, std::vector<SearchResult>& results) const {
    cv::Mat query_img = preprocessor_.loadAndPreprocess(query_image_path);
    FeatureVector query_feat = extractor_.extract(query_img);

    std::vector<std::pair<ImageRecord, FeatureVector>> rows;
    if (!repo_.fetchAllFeatures(rows) || rows.empty()) {
        return false;
    }

    results.clear();
    results.reserve(rows.size());
    for (const auto& row : rows) {
        SearchResult r;
        r.image_id = row.first.id;
        r.file_path = row.first.file_path;
        r.class_label = row.first.class_label;
        r.distance = weightedDistance(query_feat, row.second);
        results.push_back(std::move(r));
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });
    if (static_cast<int>(results.size()) > cfg_.top_k) {
        results.resize(cfg_.top_k);
    }
    return true;
}

double Searcher::weightedDistance(const FeatureVector& q, const FeatureVector& d) const {
    return cfg_.w_color * l2Distance(q.color, d.color) + cfg_.w_shape * l2Distance(q.shape, d.shape) +
           cfg_.w_texture * l2Distance(q.texture, d.texture);
}

double Searcher::l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(a[i] - b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

}  // namespace cbir

