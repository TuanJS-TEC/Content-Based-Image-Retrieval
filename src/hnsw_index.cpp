#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "hnswlib/hnswlib.h"

namespace cbir {

// ── tunables ──────────────────────────────────────────────────────────────────
static constexpr int   HNSW_DIM             = 586;  // color(512) + shape(72) + texture(2)
static constexpr int   HNSW_M               = 16;   // connections per node; higher → more RAM, better recall
static constexpr int   HNSW_EF_CONSTRUCTION = 200;  // build quality; higher → slower build, better recall

// ── Pimpl ─────────────────────────────────────────────────────────────────────
struct HnswImpl {
    hnswlib::L2Space space{static_cast<size_t>(HNSW_DIM)};
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    bool loaded = false;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
HnswIndex::HnswIndex()  : impl_(std::make_unique<HnswImpl>()) {}
HnswIndex::~HnswIndex() = default;

// ── transform ─────────────────────────────────────────────────────────────────
// Maps (color, shape, texture) → 586-dim vector where L2 distance approximates
// the original weighted metric:
//   D = w_color*Hellinger(color) + w_shape*L2(shape) + w_texture*L2(texture)
//
// Key identity: Hellinger(p,q) = L2(sqrt(p), sqrt(q)) / sqrt(2)
// So after scaling each sub-vector by sqrt(weight):
//   L2(v_q, v_d)² ≈ w_color*2*H² + w_shape*L2² + w_texture*L2²
// This preserves the ranking well enough for candidate retrieval.
std::vector<float> HnswIndex::transform(const FeatureVector& fv) {
    static const float kWColor   = std::sqrt(0.65f);
    static const float kWShape   = std::sqrt(0.25f);
    static const float kWTexture = std::sqrt(0.10f);

    std::vector<float> v;
    v.reserve(fv.color.size() + fv.shape.size() + fv.texture.size());

    for (float x : fv.color)
        v.push_back(std::sqrt(std::max(0.0f, x)) * kWColor);
    for (float x : fv.shape)
        v.push_back(x * kWShape);
    for (float x : fv.texture)
        v.push_back(x * kWTexture);

    return v;
}

// ── build ─────────────────────────────────────────────────────────────────────
bool HnswIndex::build(const std::vector<std::pair<ImageRecord, FeatureVector>>& rows) {
    if (rows.empty()) return false;

    impl_->loaded = false;
    impl_->hnsw   = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        &impl_->space,
        rows.size(),
        HNSW_M,
        HNSW_EF_CONSTRUCTION);

    int skipped = 0;
    for (const auto& [rec, feat] : rows) {
        std::vector<float> v = transform(feat);
        if (static_cast<int>(v.size()) != HNSW_DIM) { ++skipped; continue; }
        impl_->hnsw->addPoint(v.data(), static_cast<hnswlib::labeltype>(rec.id));
    }

    if (skipped > 0)
        std::cerr << "[HnswIndex] skipped " << skipped << " entries with wrong dim\n";

    impl_->loaded = true;
    return true;
}

// ── search ────────────────────────────────────────────────────────────────────
std::vector<int> HnswIndex::search(const FeatureVector& query, int k, int ef_search) const {
    if (!impl_->loaded || !impl_->hnsw || k <= 0) return {};

    std::vector<float> v = transform(query);
    if (static_cast<int>(v.size()) != HNSW_DIM) return {};

    impl_->hnsw->setEf(std::max(ef_search, k));
    auto pq = impl_->hnsw->searchKnn(v.data(), static_cast<size_t>(k));

    // searchKnn returns a max-heap (farthest first); reverse to nearest-first.
    std::vector<int> ids;
    ids.reserve(pq.size());
    while (!pq.empty()) {
        ids.push_back(static_cast<int>(pq.top().second));
        pq.pop();
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

// ── persistence ───────────────────────────────────────────────────────────────
bool HnswIndex::save(const std::string& path) const {
    if (!impl_->loaded || !impl_->hnsw) return false;
    try {
        impl_->hnsw->saveIndex(path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HnswIndex] save failed: " << e.what() << "\n";
        return false;
    }
}

bool HnswIndex::load(const std::string& path) {
    impl_->loaded = false;
    try {
        impl_->hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            &impl_->space, path);
        impl_->loaded = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HnswIndex] load failed: " << e.what() << "\n";
        impl_->hnsw.reset();
        return false;
    }
}

// ── accessors ─────────────────────────────────────────────────────────────────
bool HnswIndex::isLoaded() const {
    return impl_->loaded && impl_->hnsw != nullptr;
}

size_t HnswIndex::size() const {
    return impl_->hnsw ? impl_->hnsw->getCurrentElementCount() : 0;
}

}  // namespace cbir
