#pragma once

#include <string>
#include <vector>

#include "config.h"
#include "feature_extractor.h"
#include "image_preprocess.h"
#include "sqlite_repo.h"
#include "types.h"

namespace cbir {

class Searcher {
   public:
    Searcher(const ImagePreprocessor& preprocessor, const FeatureExtractor& extractor, SqliteRepo& repo, const Config& cfg);

    bool runQuery(const std::string& query_image_path, std::vector<SearchResult>& results) const;

   private:
    const ImagePreprocessor& preprocessor_;
    const FeatureExtractor& extractor_;
    SqliteRepo& repo_;
    const Config& cfg_;

    double weightedDistance(const FeatureVector& q, const FeatureVector& d) const;
    static double l2Distance(const std::vector<float>& a, const std::vector<float>& b);
};

}  // namespace cbir

