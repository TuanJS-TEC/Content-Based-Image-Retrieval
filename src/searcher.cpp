#include "searcher.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <opencv2/core.hpp>

#include "hnsw_index.h"

namespace cbir {

Searcher::Searcher(const ImagePreprocessor& preprocessor, const FeatureExtractor& extractor,
                   SqliteRepo& repo, const Config& cfg, const HnswIndex* ann_index)
    : preprocessor_(preprocessor), extractor_(extractor), repo_(repo), cfg_(cfg), ann_index_(ann_index) {}

bool Searcher::runQuery(const std::string& query_image_path, std::vector<SearchResult>& results) const {
    cv::Mat query_img = preprocessor_.loadAndPreprocess(query_image_path);
    FeatureVector query_feat = extractor_.extract(query_img);

    // Dataset birds all face right; query birds may face either direction.
    // min(d_orig, d_flip) per DB entry makes the search orientation-invariant
    // without needing an explicit head-direction detector.
    cv::Mat query_flipped;
    cv::flip(query_img, query_flipped, 1);
    FeatureVector query_feat_flip = extractor_.extract(query_flipped);

    if (cfg_.use_ann && ann_index_ && ann_index_->isLoaded()) {
        return runQueryAnn(query_feat, query_feat_flip, results);
    }
    return runQueryBruteForce(query_feat, query_feat_flip, results);
}

bool Searcher::runQueryBruteForce(const FeatureVector& q, const FeatureVector& q_flip,
                                   std::vector<SearchResult>& results) const {
    std::vector<std::pair<ImageRecord, FeatureVector>> rows;
    if (!repo_.fetchAllFeatures(rows) || rows.empty()) return false;

    results.clear();
    results.reserve(rows.size());
    for (const auto& row : rows) {
        SearchResult r;
        r.image_id   = row.first.id;
        r.file_path  = row.first.file_path;
        r.class_label = row.first.class_label;
        r.distance   = std::min(weightedDistance(q,      row.second),
                                weightedDistance(q_flip, row.second));
        results.push_back(std::move(r));
    }

    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) { return a.distance < b.distance; });
    if (static_cast<int>(results.size()) > cfg_.top_k)
        results.resize(cfg_.top_k);
    return true;
}

bool Searcher::runQueryAnn(const FeatureVector& q, const FeatureVector& q_flip,
                            std::vector<SearchResult>& results) const {
    const int candidates_k = cfg_.top_k * cfg_.ann_oversample;

    // Retrieve candidates for both orientations; the ANN index was built on
    // original-orientation features, so searching the flipped query exposes
    // left-facing birds that might otherwise be missed.
    std::vector<int> ids_orig = ann_index_->search(q,      candidates_k, cfg_.ann_ef_search);
    std::vector<int> ids_flip = ann_index_->search(q_flip, candidates_k, cfg_.ann_ef_search);

    // Merge deduplicating while preserving insertion order.
    std::unordered_set<int> seen;
    std::vector<int> candidate_ids;
    candidate_ids.reserve(ids_orig.size() + ids_flip.size());
    for (int id : ids_orig) { if (seen.insert(id).second) candidate_ids.push_back(id); }
    for (int id : ids_flip) { if (seen.insert(id).second) candidate_ids.push_back(id); }

    if (candidate_ids.empty()) return false;

    // Re-rank the small candidate set with the exact weighted distance.
    results.clear();
    results.reserve(candidate_ids.size());
    for (int id : candidate_ids) {
        const auto* entry = repo_.findById(id);
        if (!entry) continue;
        SearchResult r;
        r.image_id    = entry->first.id;
        r.file_path   = entry->first.file_path;
        r.class_label = entry->first.class_label;
        r.distance    = std::min(weightedDistance(q,      entry->second),
                                 weightedDistance(q_flip, entry->second));
        results.push_back(std::move(r));
    }

    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) { return a.distance < b.distance; });
    if (static_cast<int>(results.size()) > cfg_.top_k)
        results.resize(cfg_.top_k);
    return !results.empty();
}

double Searcher::weightedDistance(const FeatureVector& q, const FeatureVector& d) const {
    // Hellinger for color (better for sparse probability histograms)
    // L2 for shape and texture
    return cfg_.w_color   * hellingerDistance(q.color,   d.color)
         + cfg_.w_shape   * l2Distance(q.shape,   d.shape)
         + cfg_.w_texture * l2Distance(q.texture, d.texture);
}

double Searcher::l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = static_cast<double>(a[i] - b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double Searcher::hellingerDistance(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = std::sqrt(std::max(0.0, static_cast<double>(a[i])))
                          - std::sqrt(std::max(0.0, static_cast<double>(b[i])));
        sum += diff * diff;
    }
    return std::sqrt(sum * 0.5);
}

}  // namespace cbir

