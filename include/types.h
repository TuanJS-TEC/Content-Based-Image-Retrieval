#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace cbir {

struct FeatureVector {
    std::vector<float> color;
    std::vector<float> shape;
    std::vector<float> texture;

    std::vector<float> full() const {
        std::vector<float> out;
        out.reserve(color.size() + shape.size() + texture.size());
        out.insert(out.end(), color.begin(), color.end());
        out.insert(out.end(), shape.begin(), shape.end());
        out.insert(out.end(), texture.begin(), texture.end());
        return out;
    }
};

struct ImageRecord {
    int id = -1;
    std::string file_path;
    std::string class_label;
    int width = 0;
    int height = 0;
};

struct SearchResult {
    int image_id = -1;
    std::string file_path;
    std::string class_label;
    double distance = 0.0;
};

inline std::string vectorToCsv(const std::vector<float>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << v[i];
    }
    return oss.str();
}

inline std::vector<float> csvToVector(const std::string& s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(std::stof(token));
        }
    }
    return out;
}

}  // namespace cbir

