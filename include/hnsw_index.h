#pragma once

#include <memory>
#include <string>
#include <vector>

#include "types.h"

namespace cbir {

// Pimpl: hnswlib headers are heavy — keep them out of the public header.
struct HnswImpl;

class HnswIndex {
   public:
    HnswIndex();
    ~HnswIndex();

    // Build index from all (record, feature) pairs loaded from the DB.
    bool build(const std::vector<std::pair<ImageRecord, FeatureVector>>& rows);

    // Return up to k image_ids sorted by approximate L2 distance.
    std::vector<int> search(const FeatureVector& query, int k, int ef_search = 64) const;

    bool save(const std::string& path) const;
    bool load(const std::string& path);
    bool isLoaded() const;
    size_t size() const;

    // Convert a FeatureVector to the L2-compatible 586-dim representation used
    // internally: sqrt(color) * sqrt(w_color), shape * sqrt(w_shape), …
    // Exposed so callers (e.g. Searcher) can reuse the same transform.
    static std::vector<float> transform(const FeatureVector& fv);

   private:
    std::unique_ptr<HnswImpl> impl_;
};

}  // namespace cbir
