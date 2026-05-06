#pragma once

#include <sqlite3.h>

#include <string>
#include <vector>

#include "types.h"

namespace cbir {

class SqliteRepo {
   public:
    explicit SqliteRepo(std::string db_path);
    ~SqliteRepo();

    bool open();
    void close();
    bool initSchema();

    int upsertImage(const ImageRecord& rec);
    bool upsertFeatures(int image_id, const FeatureVector& feat, const std::string& norm_version = "v1");

    bool fetchAllFeatures(std::vector<std::pair<ImageRecord, FeatureVector>>& rows) const;

   private:
    std::string db_path_;
    sqlite3* db_ = nullptr;
};

}  // namespace cbir

