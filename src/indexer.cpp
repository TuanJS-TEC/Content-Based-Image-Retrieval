#include "indexer.h"

#include <cctype>
#include <filesystem>
#include <iostream>

namespace cbir {

namespace fs = std::filesystem;

namespace {
bool isImageFile(const fs::path& p) {
    if (!p.has_extension()) {
        return false;
    }
    std::string ext = p.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(::tolower(c));
    }
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}
}  // namespace

Indexer::Indexer(const ImagePreprocessor& preprocessor, const FeatureExtractor& extractor, SqliteRepo& repo)
    : preprocessor_(preprocessor), extractor_(extractor), repo_(repo) {}

bool Indexer::run(const std::string& dataset_path) {
    if (!fs::exists(dataset_path)) {
        std::cerr << "Dataset path does not exist: " << dataset_path << "\n";
        return false;
    }

    int processed = 0;
    int failed = 0;

    for (const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
        if (!entry.is_regular_file() || !isImageFile(entry.path())) {
            continue;
        }

        try {
            cv::Mat img = preprocessor_.loadAndPreprocess(entry.path().string());
            FeatureVector feat = extractor_.extract(img);

            ImageRecord rec;
            rec.file_path = entry.path().string();
            rec.class_label = entry.path().parent_path().filename().string();
            rec.width = img.cols;
            rec.height = img.rows;

            int image_id = repo_.upsertImage(rec);
            if (image_id < 0 || !repo_.upsertFeatures(image_id, feat)) {
                ++failed;
                continue;
            }
            ++processed;
        } catch (const std::exception& e) {
            ++failed;
            std::cerr << "[WARN] skip " << entry.path() << " reason: " << e.what() << "\n";
        }
    }

    std::cout << "Indexing done. processed=" << processed << ", failed=" << failed << "\n";
    return processed > 0;
}

}  // namespace cbir

