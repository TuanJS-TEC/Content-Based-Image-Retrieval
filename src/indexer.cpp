#include "indexer.h"

#include <cctype>
#include <filesystem>
#include <iostream>
#include <vector>

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

bool isNumericToken(const std::string& token) {
    if (token.empty()) {
        return false;
    }
    for (char c : token) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

std::string parseClassFromFileName(const fs::path& image_path) {
    std::string stem = image_path.stem().string();
    std::vector<std::string> tokens;
    std::string cur;

    for (char c : stem) {
        if (c == '_') {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        tokens.push_back(cur);
    }

    while (tokens.size() > 1 && isNumericToken(tokens.back())) {
        tokens.pop_back();
    }

    if (tokens.empty()) {
        return stem;
    }

    std::string out = tokens[0];
    for (size_t i = 1; i < tokens.size(); ++i) {
        out += "_" + tokens[i];
    }
    return out;
}

std::string inferClassLabel(const fs::path& image_path, const fs::path& dataset_root) {
    fs::path parent = image_path.parent_path();
    if (!dataset_root.empty() && parent != dataset_root && parent.has_filename()) {
        return parent.filename().string();
    }
    return parseClassFromFileName(image_path);
}
}  // namespace

Indexer::Indexer(const ImagePreprocessor& preprocessor, const FeatureExtractor& extractor, SqliteRepo& repo)
    : preprocessor_(preprocessor), extractor_(extractor), repo_(repo) {}

bool Indexer::run(const std::string& dataset_path) {
    fs::path dataset_root = fs::weakly_canonical(fs::path(dataset_path));
    if (!fs::exists(dataset_root)) {
        std::cerr << "Dataset path does not exist: " << dataset_path << "\n";
        return false;
    }

    int processed = 0;
    int failed = 0;

    for (const auto& entry : fs::recursive_directory_iterator(dataset_root)) {
        if (!entry.is_regular_file() || !isImageFile(entry.path())) {
            continue;
        }

        try {
            cv::Mat img = preprocessor_.loadAndPreprocess(entry.path().string());
            FeatureVector feat = extractor_.extract(img);

            ImageRecord rec;
            rec.file_path = entry.path().string();
            rec.class_label = inferClassLabel(entry.path(), dataset_root);
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

