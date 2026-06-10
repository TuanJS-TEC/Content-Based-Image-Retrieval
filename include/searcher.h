#pragma once

#include <string>
#include <vector>

#include "config.h"
#include "feature_extractor.h"
#include "image_preprocess.h"
#include "sqlite_repo.h"
#include "types.h"

namespace cbir {

class HnswIndex;  // forward declaration — avoids pulling hnswlib into every TU

class Searcher {
   public:
    Searcher(const ImagePreprocessor& preprocessor, const FeatureExtractor& extractor,
             SqliteRepo& repo, const Config& cfg, const HnswIndex* ann_index = nullptr);

    bool runQuery(const std::string& query_image_path, std::vector<SearchResult>& results) const;

   private:
    const ImagePreprocessor& preprocessor_;
    const FeatureExtractor& extractor_;
    SqliteRepo& repo_;
    const Config& cfg_;
    const HnswIndex* ann_index_;

    // Exact brute-force over the full memory cache.
    bool runQueryBruteForce(const FeatureVector& q, const FeatureVector& q_flip,
                             std::vector<SearchResult>& results) const;

    // ANN candidate retrieval + exact re-ranking.
    bool runQueryAnn(const FeatureVector& q, const FeatureVector& q_flip,
                     std::vector<SearchResult>& results) const;

    double weightedDistance(const FeatureVector& q, const FeatureVector& d) const;
    static double l2Distance(const std::vector<float>& a, const std::vector<float>& b);
    static double hellingerDistance(const std::vector<float>& a, const std::vector<float>& b);
};

}  // namespace cbir

