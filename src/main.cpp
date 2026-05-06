#include <iostream>
#include <string>
#include <vector>

#include "config.h"
#include "feature_extractor.h"
#include "image_preprocess.h"
#include "indexer.h"
#include "searcher.h"
#include "sqlite_repo.h"

namespace {

void printUsage() {
    std::cout << "Usage:\n"
              << "  cbir_app --mode index [--dataset <path>] [--db <path>]\n"
              << "  cbir_app --mode query --image <path> [--db <path>] [--topk <n>]\n";
}

std::string getArgValue(const std::vector<std::string>& args, const std::string& key) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    return "";
}

bool hasArg(const std::vector<std::string>& args, const std::string& key) {
    for (const auto& a : args) {
        if (a == key) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || hasArg(args, "--help")) {
        printUsage();
        return 0;
    }

    cbir::Config cfg;
    std::string mode = getArgValue(args, "--mode");
    if (mode.empty()) {
        std::cerr << "Missing --mode\n";
        printUsage();
        return 1;
    }

    std::string dataset_arg = getArgValue(args, "--dataset");
    std::string db_arg = getArgValue(args, "--db");
    std::string image_arg = getArgValue(args, "--image");
    std::string topk_arg = getArgValue(args, "--topk");

    if (!dataset_arg.empty()) {
        cfg.dataset_path = dataset_arg;
    }
    if (!db_arg.empty()) {
        cfg.db_path = db_arg;
    }
    if (!image_arg.empty()) {
        cfg.query_image_path = image_arg;
    }
    if (!topk_arg.empty()) {
        cfg.top_k = std::stoi(topk_arg);
    }

    cbir::ImagePreprocessor preprocessor(cfg.resize_width, cfg.resize_height);
    cbir::FeatureExtractor extractor(8);
    cbir::SqliteRepo repo(cfg.db_path);

    if (!repo.open() || !repo.initSchema()) {
        std::cerr << "Failed to open/init SQLite DB: " << cfg.db_path << "\n";
        return 1;
    }

    if (mode == "index") {
        cbir::Indexer indexer(preprocessor, extractor, repo);
        bool ok = indexer.run(cfg.dataset_path);
        return ok ? 0 : 1;
    }

    if (mode == "query") {
        if (cfg.query_image_path.empty()) {
            std::cerr << "Missing --image for query mode\n";
            return 1;
        }

        cbir::Searcher searcher(preprocessor, extractor, repo, cfg);
        std::vector<cbir::SearchResult> results;
        bool ok = searcher.runQuery(cfg.query_image_path, results);
        if (!ok) {
            std::cerr << "Query failed. Ensure DB has indexed features.\n";
            return 1;
        }

        std::cout << "Top-" << cfg.top_k << " results:\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::cout << (i + 1) << ". dist=" << r.distance << " label=" << r.class_label << " path=" << r.file_path
                      << "\n";
        }
        return 0;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    printUsage();
    return 1;
}

