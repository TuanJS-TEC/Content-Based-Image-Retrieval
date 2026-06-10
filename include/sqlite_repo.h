#pragma once

#include <sqlite3.h>

#include <string>
#include <unordered_map>
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

    // O(1) lookup by image_id from the in-memory cache (valid after loadAllToMemory).
    const std::pair<ImageRecord, FeatureVector>* findById(int image_id) const;

    bool loadAllToMemory();
    bool isIndexLoaded() const { return memoryCacheLoaded_; }
    int cachedCount() const { return static_cast<int>(memoryCache_.size()); }

   private:
    std::string db_path_;
    sqlite3* db_ = nullptr;
    mutable std::vector<std::pair<ImageRecord, FeatureVector>> memoryCache_;
    mutable std::unordered_map<int, size_t> id_index_;  // image_id → index in memoryCache_
    bool memoryCacheLoaded_ = false;

    bool fetchFromDb(std::vector<std::pair<ImageRecord, FeatureVector>>& rows) const;
};

}  // namespace cbir

