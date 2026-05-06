#pragma once

#include <string>

namespace cbir {

struct Config {
    std::string dataset_path = "/Users/minhtuansfile/HoangTuan_Code/CBIR/Dataset";
    std::string db_path = "bird_cbir.db";
    std::string query_image_path;
    int top_k = 5;
    int resize_width = 256;
    int resize_height = 256;

    float w_color = 0.5f;
    float w_shape = 0.3f;
    float w_texture = 0.2f;
};

}  // namespace cbir

