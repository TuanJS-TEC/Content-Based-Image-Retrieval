#pragma once

#include <string>

namespace cbir {

struct Config {
    std::string dataset_path = "/Users/minhtuansfile/HoangTuan_Code/CBIR/Dataset_resized_256";
    std::string db_path = "bird_cbir.db";
    std::string query_image_path;
    int top_k = 5;
    int resize_width = 256;
    int resize_height = 256;

    float w_color   = 0.65f;
    float w_shape   = 0.25f;
    float w_texture = 0.10f;

    bool use_ann        = true;
    int  ann_ef_search  = 64;  // HNSW search quality: higher → better recall, slower
    int  ann_oversample = 4;   // fetch top_k * oversample candidates, then re-rank exactly
};

}  // namespace cbir

