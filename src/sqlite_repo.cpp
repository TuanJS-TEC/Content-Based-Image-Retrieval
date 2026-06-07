#include "sqlite_repo.h"

#include <iostream>
#include <sstream>

namespace cbir {

namespace {
bool execSql(sqlite3* db, const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQLite exec failed: " << (err_msg ? err_msg : "unknown error") << "\n";
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}
}  // namespace

SqliteRepo::SqliteRepo(std::string db_path) : db_path_(std::move(db_path)) {}

SqliteRepo::~SqliteRepo() { close(); }

bool SqliteRepo::open() {
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    return rc == SQLITE_OK && db_ != nullptr;
}

void SqliteRepo::close() {
    if (db_ != nullptr) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool SqliteRepo::initSchema() {
    const char* images_sql =
        "CREATE TABLE IF NOT EXISTS images ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "file_path TEXT NOT NULL UNIQUE,"
        "class_label TEXT,"
        "width INTEGER,"
        "height INTEGER,"
        "created_at TEXT DEFAULT CURRENT_TIMESTAMP"
        ");";

    const char* features_sql =
        "CREATE TABLE IF NOT EXISTS features ("
        "image_id INTEGER PRIMARY KEY,"
        "color_vec TEXT NOT NULL,"
        "shape_vec TEXT NOT NULL,"
        "texture_vec TEXT NOT NULL,"
        "full_vec TEXT NOT NULL,"
        "norm_version TEXT NOT NULL,"
        "FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE"
        ");";

    const char* index_sql = "CREATE INDEX IF NOT EXISTS idx_images_label ON images(class_label);";

    return execSql(db_, images_sql) && execSql(db_, features_sql) && execSql(db_, index_sql);
}

int SqliteRepo::upsertImage(const ImageRecord& rec) {
    const char* sql =
        "INSERT INTO images(file_path, class_label, width, height) VALUES(?,?,?,?) "
        "ON CONFLICT(file_path) DO UPDATE SET "
        "class_label=excluded.class_label, width=excluded.width, height=excluded.height;";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }

    sqlite3_bind_text(stmt, 1, rec.file_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, rec.class_label.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, rec.width);
    sqlite3_bind_int(stmt, 4, rec.height);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);

    const char* id_sql = "SELECT id FROM images WHERE file_path = ?;";
    if (sqlite3_prepare_v2(db_, id_sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }
    sqlite3_bind_text(stmt, 1, rec.file_path.c_str(), -1, SQLITE_TRANSIENT);

    int image_id = -1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        image_id = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return image_id;
}

bool SqliteRepo::upsertFeatures(int image_id, const FeatureVector& feat, const std::string& norm_version) {
    const char* sql =
        "INSERT INTO features(image_id, color_vec, shape_vec, texture_vec, full_vec, norm_version) "
        "VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(image_id) DO UPDATE SET "
        "color_vec=excluded.color_vec, "
        "shape_vec=excluded.shape_vec, "
        "texture_vec=excluded.texture_vec, "
        "full_vec=excluded.full_vec, "
        "norm_version=excluded.norm_version;";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    std::string color = vectorToCsv(feat.color);
    std::string shape = vectorToCsv(feat.shape);
    std::string texture = vectorToCsv(feat.texture);
    std::string full = vectorToCsv(feat.full());

    sqlite3_bind_int(stmt, 1, image_id);
    sqlite3_bind_text(stmt, 2, color.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, shape.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, texture.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, full.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, norm_version.c_str(), -1, SQLITE_TRANSIENT);

    bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return ok;
}

bool SqliteRepo::fetchFromDb(std::vector<std::pair<ImageRecord, FeatureVector>>& rows) const {
    const char* sql =
        "SELECT i.id, i.file_path, i.class_label, i.width, i.height, "
        "f.color_vec, f.shape_vec, f.texture_vec "
        "FROM images i JOIN features f ON i.id = f.image_id;";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ImageRecord rec;
        rec.id = sqlite3_column_int(stmt, 0);
        rec.file_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const unsigned char* raw_label = sqlite3_column_text(stmt, 2);
        rec.class_label = raw_label ? reinterpret_cast<const char*>(raw_label) : "";
        rec.width = sqlite3_column_int(stmt, 3);
        rec.height = sqlite3_column_int(stmt, 4);

        FeatureVector feat;
        feat.color = csvToVector(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        feat.shape = csvToVector(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6)));
        feat.texture = csvToVector(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7)));

        rows.push_back({rec, feat});
    }
    sqlite3_finalize(stmt);
    return true;
}

bool SqliteRepo::fetchAllFeatures(std::vector<std::pair<ImageRecord, FeatureVector>>& rows) const {
    if (memoryCacheLoaded_) {
        rows = memoryCache_;
        return true;
    }
    return fetchFromDb(rows);
}

bool SqliteRepo::loadAllToMemory() {
    memoryCacheLoaded_ = false;
    memoryCache_.clear();
    if (!fetchFromDb(memoryCache_)) return false;
    memoryCacheLoaded_ = true;
    return true;
}

}  // namespace cbir

