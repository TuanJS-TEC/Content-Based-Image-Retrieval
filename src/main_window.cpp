#include "main_window.h"

#include <cmath>

#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QMessageBox>
#include <QPushButton>
#include <QSizePolicy>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "config.h"
#include "feature_extractor.h"
#include "hnsw_index.h"
#include "image_preprocess.h"
#include "indexer.h"
#include "searcher.h"
#include "sqlite_repo.h"

namespace cbir {
namespace {

cv::Mat buildForegroundMaskForUi(const cv::Mat& imageBgr, QString& log) {
    const int rows = imageBgr.rows;
    const int cols = imageBgr.cols;
    const int cx = cols / 2;
    const int cy = rows / 2;

    // Estimate background color from border pixels in LAB space
    cv::Mat lab;
    cv::cvtColor(imageBgr, lab, cv::COLOR_BGR2Lab);

    log += QString("  Image: %1x%2 (%3 px total)\n").arg(cols).arg(rows).arg(cols * rows);

    const int border = std::max(8, std::min(rows, cols) / 20);
    float sumL = 0, sumA = 0, sumB = 0;
    int cnt = 0;
    for (int r = 0; r < rows; r += 2) {
        for (int c = 0; c < cols; c += 2) {
            if (r < border || r >= rows - border || c < border || c >= cols - border) {
                const cv::Vec3b p = lab.at<cv::Vec3b>(r, c);
                sumL += p[0]; sumA += p[1]; sumB += p[2];
                ++cnt;
            }
        }
    }
    if (cnt < 1) cnt = 1;
    const float bgL = sumL / cnt, bgA = sumA / cnt, bgB = sumB / cnt;

    float varL = 0, varA = 0, varB = 0;
    for (int r = 0; r < rows; r += 2) {
        for (int c = 0; c < cols; c += 2) {
            if (r < border || r >= rows - border || c < border || c >= cols - border) {
                const cv::Vec3b p = lab.at<cv::Vec3b>(r, c);
                varL += (p[0] - bgL) * (p[0] - bgL);
                varA += (p[1] - bgA) * (p[1] - bgA);
                varB += (p[2] - bgB) * (p[2] - bgB);
            }
        }
    }
    const float sigL = std::max(5.0f, std::sqrt(varL / cnt));
    const float sigA = std::max(5.0f, std::sqrt(varA / cnt));
    const float sigB = std::max(5.0f, std::sqrt(varB / cnt));

    log += QString("  BG(LAB): L=%1±%2  a=%3±%4  b=%5±%6  (%7 border px sampled)\n")
               .arg(bgL, 0, 'f', 1).arg(sigL, 0, 'f', 1)
               .arg(bgA, 0, 'f', 1).arg(sigA, 0, 'f', 1)
               .arg(bgB, 0, 'f', 1).arg(sigB, 0, 'f', 1)
               .arg(cnt);

    cv::Mat colorMask(imageBgr.size(), CV_8U, cv::Scalar(0));
    const float mahThr2 = 2.0f * 2.0f;
    for (int r = 0; r < rows; ++r) {
        uint8_t* mptr = colorMask.ptr<uint8_t>(r);
        const cv::Vec3b* lptr = lab.ptr<cv::Vec3b>(r);
        for (int c = 0; c < cols; ++c) {
            const float dL = (lptr[c][0] - bgL) / sigL;
            const float da = (lptr[c][1] - bgA) / sigA;
            const float db = (lptr[c][2] - bgB) / sigB;
            if (dL * dL + da * da + db * db > mahThr2) {
                mptr[c] = 255;
            }
        }
    }
    const int colorFgPx = cv::countNonZero(colorMask);
    log += QString("  Color mask (2.0σ Mahalanobis): %1 px  (%2%)\n")
               .arg(colorFgPx)
               .arg(100.0 * colorFgPx / (cols * rows), 0, 'f', 1);

    cv::Mat gray;
    cv::cvtColor(imageBgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat otsuMask;
    const double otsuThr = cv::threshold(gray, otsuMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    const int checkR = std::min(rows, cols) / 5;
    cv::Mat checkCircle(gray.size(), CV_8U, cv::Scalar(0));
    cv::circle(checkCircle, cv::Point(cx, cy), checkR, cv::Scalar(255), cv::FILLED);
    const int centerArea = cv::countNonZero(checkCircle);
    cv::Mat centerOtsu;
    cv::bitwise_and(otsuMask, checkCircle, centerOtsu);
    const int centerFgCnt = cv::countNonZero(centerOtsu);
    const bool inverted = (centerFgCnt < centerArea / 3);
    if (inverted) {
        cv::bitwise_not(otsuMask, otsuMask);
    }
    log += QString("  Otsu thr=%1 | center fg=%2/%3 (%4%) → mask %5\n")
               .arg(static_cast<int>(otsuThr))
               .arg(centerFgCnt).arg(centerArea)
               .arg(100.0 * centerFgCnt / centerArea, 0, 'f', 1)
               .arg(inverted ? "INVERTED (dark bird)" : "kept as-is");

    cv::Mat centerPrior(imageBgr.size(), CV_8U, cv::Scalar(0));
    const int rx = static_cast<int>(cols * 0.45);
    const int ry = static_cast<int>(rows * 0.45);
    cv::ellipse(centerPrior, cv::Point(cx, cy), cv::Size(rx, ry), 0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED);

    cv::Mat centeredColor, centeredOtsu;
    cv::bitwise_and(colorMask, centerPrior, centeredColor);
    cv::bitwise_and(otsuMask, centerPrior, centeredOtsu);

    cv::Mat candidate;
    QString candidateSrc;
    if (cv::countNonZero(centeredColor) >= 400) {
        candidate = centeredColor;
        candidateSrc = QString("color (%1 px inside ellipse %2x%3)").arg(cv::countNonZero(centeredColor)).arg(rx).arg(ry);
    } else if (cv::countNonZero(centeredOtsu) >= 400) {
        candidate = centeredOtsu;
        candidateSrc = QString("otsu (%1 px inside ellipse %2x%3)").arg(cv::countNonZero(centeredOtsu)).arg(rx).arg(ry);
    } else {
        candidate = colorMask;
        candidateSrc = QString("color-full (%1 px, no prior)").arg(cv::countNonZero(colorMask));
    }
    log += QString("  Candidate: %1\n").arg(candidateSrc);

    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17)));
    const int afterClose17 = cv::countNonZero(candidate);
    cv::morphologyEx(candidate, candidate, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    const int afterOpen5 = cv::countNonZero(candidate);
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    const int afterClose9 = cv::countNonZero(candidate);
    log += QString("  Morphology: CLOSE(17)=%1px → OPEN(5)=%2px → CLOSE(9)=%3px\n")
               .arg(afterClose17).arg(afterOpen5).arg(afterClose9);

    // Inline connected-component noise filter with count logging
    cv::Mat labeled, ccStats, ccCentroids;
    const int nComp = cv::connectedComponentsWithStats(candidate, labeled, ccStats, ccCentroids, 8, CV_32S);
    cv::Mat result = cv::Mat::zeros(candidate.size(), CV_8U);
    int removedBlobs = 0;
    for (int i = 1; i < nComp; ++i) {
        if (ccStats.at<int>(i, cv::CC_STAT_AREA) >= 200) {
            result.setTo(255, labeled == i);
        } else {
            ++removedBlobs;
        }
    }
    const int finalPx = cv::countNonZero(result);
    log += QString("  Noise filter: removed %1 blob(s) < 200px | Final mask: %2 px (%3%)\n")
               .arg(removedBlobs).arg(finalPx)
               .arg(100.0 * finalPx / (cols * rows), 0, 'f', 1);

    return result;
}

float centerWeight(int r, int c, int rows, int cols) {
    const float cy = static_cast<float>(rows - 1) * 0.5f;
    const float cx = static_cast<float>(cols - 1) * 0.5f;
    const float dy = (static_cast<float>(r) - cy) / std::max(1.0f, static_cast<float>(rows));
    const float dx = (static_cast<float>(c) - cx) / std::max(1.0f, static_cast<float>(cols));
    const float sigma = 0.25f;
    return std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
}

QPixmap matToPixmap(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    return QPixmap::fromImage(img.copy());
}

static std::string hnswIndexPath(const std::string& db_path) {
    const auto pos = db_path.rfind('.');
    return (pos != std::string::npos ? db_path.substr(0, pos) : db_path) + ".hnsw";
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Bird CBIR - Qt GUI");
    resize(1380, 860);

    auto* central = new QWidget(this);
    auto* rootLayout = new QVBoxLayout(central);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(6);

    // ── 1. Compact input bar (2-row grid) ────────────────────────────
    auto* inputGroup = new QGroupBox("Input", central);
    auto* inputGrid = new QGridLayout(inputGroup);
    inputGrid->setHorizontalSpacing(6);
    inputGrid->setVerticalSpacing(4);

    datasetPathEdit_ = new QLineEdit(inputGroup);
    datasetPathEdit_->setText(QString::fromStdString(Config().dataset_path));
    auto* datasetPickBtn = new QPushButton("Browse...", inputGroup);
    inputGrid->addWidget(new QLabel("Dataset:", inputGroup), 0, 0);
    inputGrid->addWidget(datasetPathEdit_, 0, 1);
    inputGrid->addWidget(datasetPickBtn, 0, 2);

    dbPathEdit_ = new QLineEdit(inputGroup);
    dbPathEdit_->setText(QString::fromStdString(Config().db_path));
    auto* dbPickBtn = new QPushButton("Browse...", inputGroup);
    inputGrid->addWidget(new QLabel("SQLite DB:", inputGroup), 0, 3);
    inputGrid->addWidget(dbPathEdit_, 0, 4);
    inputGrid->addWidget(dbPickBtn, 0, 5);

    queryPathEdit_ = new QLineEdit(inputGroup);
    auto* queryPickBtn = new QPushButton("Browse...", inputGroup);
    inputGrid->addWidget(new QLabel("Query image:", inputGroup), 0, 6);
    inputGrid->addWidget(queryPathEdit_, 0, 7);
    inputGrid->addWidget(queryPickBtn, 0, 8);

    topKEdit_ = new QLineEdit(inputGroup);
    topKEdit_->setText("5");
    topKEdit_->setMaximumWidth(60);
    inputGrid->addWidget(new QLabel("Top-K:", inputGroup), 1, 0);
    inputGrid->addWidget(topKEdit_, 1, 1);

    searchButton_ = new QPushButton("Search Top-K", inputGroup);
    inputGrid->addWidget(searchButton_, 1, 8);

    inputGrid->setColumnStretch(1, 2);
    inputGrid->setColumnStretch(4, 2);
    inputGrid->setColumnStretch(7, 3);

    rootLayout->addWidget(inputGroup);

    // ── 2. Main content: query image (left) | pipeline+top5 (right) ──
    auto* mainSplitter = new QSplitter(Qt::Horizontal, central);

    // Left: query image (tall)
    auto* queryGroup = new QGroupBox("Query image", mainSplitter);
    auto* queryVL = new QVBoxLayout(queryGroup);
    queryPreviewLabel_ = new QLabel(queryGroup);
    queryPreviewLabel_->setMinimumSize(280, 360);
    queryPreviewLabel_->setAlignment(Qt::AlignCenter);
    queryPreviewLabel_->setStyleSheet("border: 1px solid #999;");
    queryPreviewLabel_->setText("No query image");
    queryPreviewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    queryVL->addWidget(queryPreviewLabel_);
    mainSplitter->addWidget(queryGroup);

    // Right: pipeline steps + top-5 results
    auto* rightWidget = new QWidget(mainSplitter);
    auto* rightVL = new QVBoxLayout(rightWidget);
    rightVL->setContentsMargins(0, 0, 0, 0);
    rightVL->setSpacing(6);

    auto* pipelineGroup = new QGroupBox("Input processing pipeline", rightWidget);
    auto* pipelineHL = new QHBoxLayout(pipelineGroup);
    pipelineHL->setSpacing(4);
    for (int i = 0; i < 5; ++i) {
        auto* card = new QWidget(pipelineGroup);
        auto* cardVL = new QVBoxLayout(card);
        cardVL->setContentsMargins(4, 4, 4, 4);
        auto* imgLbl = new QLabel(card);
        imgLbl->setAlignment(Qt::AlignCenter);
        imgLbl->setMinimumSize(190, 150);
        imgLbl->setStyleSheet("border: 1px solid #999;");
        imgLbl->setText(QString("Step %1").arg(i + 1));
        imgLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        auto* txtLbl = new QLabel(card);
        txtLbl->setAlignment(Qt::AlignCenter);
        txtLbl->setWordWrap(true);
        txtLbl->setText("-");
        cardVL->addWidget(imgLbl);
        cardVL->addWidget(txtLbl);
        pipelineHL->addWidget(card, 1);
        pipelineImageLabels_[i] = imgLbl;
        pipelineTextLabels_[i] = txtLbl;
    }
    rightVL->addWidget(pipelineGroup, 3);

    auto* top5Group = new QGroupBox("Top-5 matched birds", rightWidget);
    auto* top5HL = new QHBoxLayout(top5Group);
    top5HL->setSpacing(4);
    for (int i = 0; i < 5; ++i) {
        auto* card = new QWidget(top5Group);
        auto* cardVL = new QVBoxLayout(card);
        cardVL->setContentsMargins(4, 4, 4, 4);
        auto* imgLbl = new QLabel(card);
        imgLbl->setAlignment(Qt::AlignCenter);
        imgLbl->setMinimumSize(170, 130);
        imgLbl->setStyleSheet("border: 1px solid #999;");
        imgLbl->setText(QString("Top %1").arg(i + 1));
        imgLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        auto* infoLbl = new QLabel(card);
        infoLbl->setAlignment(Qt::AlignCenter);
        infoLbl->setWordWrap(true);
        infoLbl->setText("-");
        cardVL->addWidget(imgLbl);
        cardVL->addWidget(infoLbl);
        top5HL->addWidget(card, 1);
        topImageLabels_[i] = imgLbl;
        topInfoLabels_[i] = infoLbl;
    }
    rightVL->addWidget(top5Group, 2);

    mainSplitter->addWidget(rightWidget);
    mainSplitter->setStretchFactor(0, 1);   // query image: 1 part
    mainSplitter->setStretchFactor(1, 3);   // right panel: 3 parts
    rootLayout->addWidget(mainSplitter, 4);

    // ── 3. Bottom: search results (left) | logs (right) ───────────────
    auto* bottomSplitter = new QSplitter(Qt::Horizontal, central);

    auto* resultsGroup = new QGroupBox("Search results", bottomSplitter);
    auto* resultsVL = new QVBoxLayout(resultsGroup);
    resultTable_ = new QTableWidget(0, 4, resultsGroup);
    resultTable_->setHorizontalHeaderLabels({"Rank", "Distance", "Label", "Image Path"});
    resultTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    resultTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    resultTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    resultTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    resultTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    resultTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    resultsVL->addWidget(resultTable_);
    bottomSplitter->addWidget(resultsGroup);

    auto* logsGroup = new QGroupBox("Logs", bottomSplitter);
    auto* logsVL = new QVBoxLayout(logsGroup);
    logView_ = new QTextEdit(logsGroup);
    logView_->setReadOnly(true);
    logView_->setFont(QFont("Menlo", 10));
    logsVL->addWidget(logView_);
    bottomSplitter->addWidget(logsGroup);

    bottomSplitter->setStretchFactor(0, 3);  // table: 3 parts
    bottomSplitter->setStretchFactor(1, 2);  // logs: 2 parts
    rootLayout->addWidget(bottomSplitter, 2);

    setCentralWidget(central);

    connect(datasetPickBtn, &QPushButton::clicked, this, &MainWindow::pickDatasetFolder);
    connect(dbPickBtn, &QPushButton::clicked, this, &MainWindow::pickDbFile);
    connect(queryPickBtn, &QPushButton::clicked, this, &MainWindow::pickQueryImage);
    connect(searchButton_, &QPushButton::clicked, this, &MainWindow::runSearch);
    connect(dbPathEdit_, &QLineEdit::editingFinished, this, &MainWindow::loadIndexToRam);

    QTimer::singleShot(0, this, &MainWindow::loadIndexToRam);
}

MainWindow::~MainWindow() = default;

void MainWindow::pickDatasetFolder() {
    const QString dir = QFileDialog::getExistingDirectory(this, "Select Dataset Folder", datasetPathEdit_->text());
    if (!dir.isEmpty()) {
        datasetPathEdit_->setText(dir);
    }
}

void MainWindow::pickDbFile() {
    const QString file = QFileDialog::getSaveFileName(this, "Select SQLite DB", dbPathEdit_->text(), "DB files (*.db)");
    if (!file.isEmpty()) {
        dbPathEdit_->setText(file);
        loadIndexToRam();
    }
}

void MainWindow::loadIndexToRam() {
    const std::string dbPath = dbPathEdit_->text().toStdString();
    if (dbPath == loadedDbPath_ && persistentRepo_ && persistentRepo_->isIndexLoaded()) {
        return;
    }

    persistentRepo_ = std::make_unique<SqliteRepo>(dbPath);
    loadedDbPath_ = dbPath;

    if (!persistentRepo_->open() || !persistentRepo_->initSchema()) {
        appendLog("Index auto-load: cannot open DB: " + dbPathEdit_->text());
        return;
    }

    if (!persistentRepo_->loadAllToMemory() || persistentRepo_->cachedCount() == 0) {
        appendLog("Index empty – starting auto-indexing from: " + datasetPathEdit_->text());
        runIndexing(true);
        return;
    }

    appendLog(QString("Index loaded: %1 entries from %2")
                  .arg(persistentRepo_->cachedCount())
                  .arg(dbPathEdit_->text()));

    annIndex_ = std::make_unique<HnswIndex>();
    if (annIndex_->load(hnswIndexPath(dbPath))) {
        appendLog(QString("ANN index loaded: %1 entries").arg(static_cast<int>(annIndex_->size())));
    } else {
        appendLog("ANN index not found – will be built on next indexing.");
        annIndex_.reset();
    }
}

void MainWindow::pickQueryImage() {
    const QString file = QFileDialog::getOpenFileName(this, "Select Query Image", queryPathEdit_->text(),
                                                      "Images (*.jpg *.jpeg *.png *.bmp)");
    if (!file.isEmpty()) {
        queryPathEdit_->setText(file);
        setPreviewImage(queryPreviewLabel_, file, 300, 220);
        updatePipelineVisualization(file);
    }
}

void MainWindow::runIndexing(bool silent) {
    Config cfg;
    cfg.dataset_path = datasetPathEdit_->text().toStdString();
    cfg.db_path = dbPathEdit_->text().toStdString();

    ImagePreprocessor preprocessor(cfg.resize_width, cfg.resize_height);
    FeatureExtractor extractor(8);
    SqliteRepo repo(cfg.db_path);

    if (!repo.open() || !repo.initSchema()) {
        if (!silent) {
            QMessageBox::critical(this, "Error", "Cannot open/init SQLite database.");
        } else {
            appendLog("Auto-index: cannot open/init DB – check dataset and DB paths.");
        }
        return;
    }

    appendLog("Indexing: " + QString::fromStdString(cfg.dataset_path));
    setEnabled(false);

    Indexer indexer(preprocessor, extractor, repo);
    const bool ok = indexer.run(cfg.dataset_path);

    setEnabled(true);
    if (ok) {
        appendLog("Indexing completed.");
        loadedDbPath_.clear();
        loadIndexToRam();
        buildAndSaveAnnIndex(cfg.db_path);
        if (!silent) {
            QMessageBox::information(this, "Done", "Indexing completed successfully.");
        }
    } else {
        appendLog("Indexing failed.");
        if (!silent) {
            QMessageBox::warning(this, "Failed", "Indexing failed. See logs for details.");
        }
    }
}

void MainWindow::runSearch() {
    if (queryPathEdit_->text().isEmpty()) {
        QMessageBox::warning(this, "Missing input", "Please select a query image.");
        return;
    }

    Config cfg;
    cfg.db_path = dbPathEdit_->text().toStdString();
    cfg.query_image_path = queryPathEdit_->text().toStdString();
    cfg.top_k = topKEdit_->text().toInt();
    if (cfg.top_k <= 0) {
        cfg.top_k = 5;
    }
    setPreviewImage(queryPreviewLabel_, QString::fromStdString(cfg.query_image_path), 300, 220);

    if (!persistentRepo_ || !persistentRepo_->isIndexLoaded() || loadedDbPath_ != cfg.db_path) {
        loadIndexToRam();
    }
    if (!persistentRepo_ || !persistentRepo_->isIndexLoaded()) {
        QMessageBox::warning(this, "Index not ready", "No indexed features found. Please run indexing first.");
        return;
    }

    ImagePreprocessor preprocessor(cfg.resize_width, cfg.resize_height);
    FeatureExtractor extractor(8);

    const bool usingAnn = cfg.use_ann && annIndex_ && annIndex_->isLoaded();
    appendLog(usingAnn ? "Search mode: ANN (HNSW)" : "Search mode: Brute-force");
    Searcher searcher(preprocessor, extractor, *persistentRepo_, cfg, annIndex_.get());
    std::vector<SearchResult> results;
    const bool ok = searcher.runQuery(cfg.query_image_path, results);
    if (!ok) {
        appendLog("Query failed for: " + queryPathEdit_->text());
        QMessageBox::warning(this, "Failed", "Query failed. Ensure DB has indexed features.");
        return;
    }

    resultTable_->setRowCount(static_cast<int>(results.size()));
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        const auto& r = results[i];
        resultTable_->setItem(i, 0, new QTableWidgetItem(QString::number(i + 1)));
        resultTable_->setItem(i, 1, new QTableWidgetItem(QString::number(r.distance, 'f', 6)));
        resultTable_->setItem(i, 2, new QTableWidgetItem(QString::fromStdString(r.class_label)));
        resultTable_->setItem(i, 3, new QTableWidgetItem(QString::fromStdString(r.file_path)));
    }

    for (int i = 0; i < 5; ++i) {
        if (i < static_cast<int>(results.size())) {
            const auto& r = results[i];
            setPreviewImage(topImageLabels_[i], QString::fromStdString(r.file_path), 150, 110);
            topInfoLabels_[i]->setText(QString("%1\nD=%2")
                                           .arg(QString::fromStdString(r.class_label))
                                           .arg(QString::number(r.distance, 'f', 4)));
        } else {
            topImageLabels_[i]->setPixmap(QPixmap());
            topImageLabels_[i]->setText(QString("Top %1").arg(i + 1));
            topInfoLabels_[i]->setText("-");
        }
    }
    appendLog("Query done for: " + queryPathEdit_->text() + " | topk=" + QString::number(cfg.top_k));
}

void MainWindow::appendLog(const QString& message) {
    logView_->append(message);
}

void MainWindow::setPreviewImage(QLabel* target, const QString& imagePath, int width, int height) {
    const QPixmap pix(imagePath);
    if (pix.isNull()) {
        target->setPixmap(QPixmap());
        target->setText("Cannot load image");
        return;
    }
    target->setText("");
    // Use the label's actual size if already laid out; fall back to caller-supplied dims
    const int w = target->width() > 20 ? target->width() - 4 : width;
    const int h = target->height() > 20 ? target->height() - 4 : height;
    target->setPixmap(pix.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::updatePipelineVisualization(const QString& imagePath) {
    cv::Mat raw = cv::imread(imagePath.toStdString(), cv::IMREAD_COLOR);
    if (raw.empty()) {
        for (int i = 0; i < 5; ++i) {
            pipelineImageLabels_[i]->setPixmap(QPixmap());
            pipelineImageLabels_[i]->setText("N/A");
            pipelineTextLabels_[i]->setText("Cannot load image");
        }
        return;
    }

    cv::Mat resized;
    cv::resize(raw, resized, cv::Size(256, 256), 0.0, 0.0, cv::INTER_AREA);

    appendLog("── Pipeline: " + QFileInfo(imagePath).fileName() + " ──────────────────");

    // Step 2: Lab-L channel
    cv::Mat lab;
    cv::cvtColor(resized, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);
    cv::Mat lChannelBgr;
    cv::cvtColor(labChannels[0], lChannelBgr, cv::COLOR_GRAY2BGR);

    cv::Scalar lMean, lStd;
    cv::meanStdDev(labChannels[0], lMean, lStd);
    double lMin, lMax;
    cv::minMaxLoc(labChannels[0], &lMin, &lMax);
    appendLog(QString("[Lab-L] mean=%1  std=%2  min=%3  max=%4")
                  .arg(lMean[0], 0, 'f', 1).arg(lStd[0], 0, 'f', 1)
                  .arg(static_cast<int>(lMin)).arg(static_cast<int>(lMax)));

    // Step 3: Foreground mask
    appendLog("[Mask] Building foreground mask...");
    QString maskLog;
    cv::Mat mask = buildForegroundMaskForUi(resized, maskLog);
    appendLog(maskLog.trimmed());

    cv::Mat maskBgr;
    cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);

    // Step 4: Foreground only
    cv::Mat foregroundOnly;
    cv::bitwise_and(resized, resized, foregroundOnly, mask);
    const int fgPx = cv::countNonZero(mask);
    appendLog(QString("[FG-only] Masked pixels kept: %1 / %2  (%3%)")
                  .arg(fgPx).arg(resized.rows * resized.cols)
                  .arg(100.0 * fgPx / (resized.rows * resized.cols), 0, 'f', 1));

    // Step 5: PWH weight map
    cv::Mat weightMap(resized.size(), CV_32F, cv::Scalar(0));
    float wFgSum = 0, wBgSum = 0;
    float wMin = 1e9f, wMax = -1e9f;
    int nFgW = 0, nBgW = 0;
    for (int r = 0; r < resized.rows; ++r) {
        float* wptr = weightMap.ptr<float>(r);
        const uint8_t* mptr = mask.ptr<uint8_t>(r);
        for (int c = 0; c < resized.cols; ++c) {
            float w = 0.25f + centerWeight(r, c, resized.rows, resized.cols);
            if (mptr[c] > 0) {
                w *= 1.8f;
                wFgSum += w; ++nFgW;
            } else {
                wBgSum += w; ++nBgW;
            }
            wptr[c] = w;
            if (w < wMin) wMin = w;
            if (w > wMax) wMax = w;
        }
    }
    const float wFgMean = nFgW > 0 ? wFgSum / nFgW : 0.0f;
    const float wBgMean = nBgW > 0 ? wBgSum / nBgW : 0.0f;
    appendLog(QString("[Weight] range=[%1, %2] | FG(%3px) mean=%4 | BG(%5px) mean=%6 | FG/BG ratio=%7")
                  .arg(wMin, 0, 'f', 3).arg(wMax, 0, 'f', 3)
                  .arg(nFgW).arg(wFgMean, 0, 'f', 3)
                  .arg(nBgW).arg(wBgMean, 0, 'f', 3)
                  .arg(nBgW > 0 ? wFgMean / wBgMean : 0.0f, 0, 'f', 2));

    cv::Mat weightNorm, weightColor;
    cv::normalize(weightMap, weightNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(weightNorm, weightColor, cv::COLORMAP_JET);

    const std::array<cv::Mat, 5> stepImages = {resized, lChannelBgr, maskBgr, foregroundOnly, weightColor};
    const std::array<QString, 5> stepNames = {QString("1) Resize 256x256"), QString("2) Lab-L channel"),
                                              QString("3) Foreground mask"), QString("4) Foreground only"),
                                              QString("5) PWH weight map")};

    for (int i = 0; i < 5; ++i) {
        auto* lbl = pipelineImageLabels_[i];
        const int pw = lbl->width() > 20 ? lbl->width() - 4 : 190;
        const int ph = lbl->height() > 20 ? lbl->height() - 4 : 150;
        lbl->setText("");
        lbl->setPixmap(matToPixmap(stepImages[i]).scaled(pw, ph, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        pipelineTextLabels_[i]->setText(stepNames[i]);
    }
}

void MainWindow::buildAndSaveAnnIndex(const std::string& db_path) {
    if (!persistentRepo_ || !persistentRepo_->isIndexLoaded()) return;

    std::vector<std::pair<ImageRecord, FeatureVector>> rows;
    if (!persistentRepo_->fetchAllFeatures(rows) || rows.empty()) return;

    appendLog(QString("Building ANN index for %1 entries…").arg(static_cast<int>(rows.size())));
    annIndex_ = std::make_unique<HnswIndex>();
    if (!annIndex_->build(rows)) {
        appendLog("ANN index build failed.");
        annIndex_.reset();
        return;
    }

    const std::string hnsw_path = hnswIndexPath(db_path);
    if (annIndex_->save(hnsw_path)) {
        appendLog(QString("ANN index saved: %1 entries → %2")
                      .arg(static_cast<int>(annIndex_->size()))
                      .arg(QString::fromStdString(hnsw_path)));
    } else {
        appendLog("ANN index save failed – index still active in memory.");
    }
}

}  // namespace cbir
