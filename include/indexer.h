#pragma once

#include <string>

#include "feature_extractor.h"
#include "image_preprocess.h"
#include "sqlite_repo.h"

namespace cbir {

class Indexer {
   public:
    Indexer(const ImagePreprocessor& preprocessor, const FeatureExtractor& extractor, SqliteRepo& repo);

    bool run(const std::string& dataset_path);

   private:
    const ImagePreprocessor& preprocessor_;
    const FeatureExtractor& extractor_;
    SqliteRepo& repo_;
};

}  // namespace cbir

