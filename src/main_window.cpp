#include "main_window.h"

#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QMessageBox>
#include <QPushButton>
#include <QSizePolicy>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "config.h"
#include "feature_extractor.h"
#include "image_preprocess.h"
#include "indexer.h"
#include "searcher.h"
#include "sqlite_repo.h"

namespace cbir {
namespace {

cv::Mat buildForegroundMaskForUi(const cv::Mat& imageBgr) {
    cv::Mat gray;
    cv::cvtColor(imageBgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat globalMask;
    cv::threshold(gray, globalMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat centerPrior(gray.size(), CV_8U, cv::Scalar(0));
    const int cx = gray.cols / 2;
    const int cy = gray.rows / 2;
    const int rx = static_cast<int>(gray.cols * 0.35);
    const int ry = static_cast<int>(gray.rows * 0.35);
    cv::ellipse(centerPrior, cv::Point(cx, cy), cv::Size(rx, ry), 0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED);

    cv::Mat centeredMask;
    cv::bitwise_and(globalMask, centerPrior, centeredMask);
    if (cv::countNonZero(centeredMask) < 200) {
        centeredMask = globalMask;
    }
    cv::morphologyEx(centeredMask, centeredMask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::morphologyEx(centeredMask, centeredMask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    return centeredMask;
}

float centerWeight(int r, int c, int rows, int cols) {
    const float cy = static_cast<float>(rows - 1) * 0.5f;
    const float cx = static_cast<float>(cols - 1) * 0.5f;
    const float dy = (static_cast<float>(r) - cy) / std::max(1.0f, static_cast<float>(rows));
    const float dx = (static_cast<float>(c) - cx) / std::max(1.0f, static_cast<float>(cols));
    const float sigma = 0.22f;
    return std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
}

QPixmap matToPixmap(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    return QPixmap::fromImage(img.copy());
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Bird CBIR - Qt GUI");
    resize(980, 680);

    auto* central = new QWidget(this);
    auto* rootLayout = new QVBoxLayout(central);

    auto* inputGroup = new QGroupBox("Input", central);
    auto* formLayout = new QFormLayout(inputGroup);

    datasetPathEdit_ = new QLineEdit(inputGroup);
    datasetPathEdit_->setText(QString::fromStdString(Config().dataset_path));
    auto* datasetRow = new QWidget(inputGroup);
    auto* datasetLayout = new QHBoxLayout(datasetRow);
    datasetLayout->setContentsMargins(0, 0, 0, 0);
    datasetLayout->addWidget(datasetPathEdit_);
    auto* datasetPickBtn = new QPushButton("Browse...", datasetRow);
    datasetLayout->addWidget(datasetPickBtn);
    formLayout->addRow("Dataset folder:", datasetRow);

    dbPathEdit_ = new QLineEdit(inputGroup);
    dbPathEdit_->setText(QString::fromStdString(Config().db_path));
    auto* dbRow = new QWidget(inputGroup);
    auto* dbLayout = new QHBoxLayout(dbRow);
    dbLayout->setContentsMargins(0, 0, 0, 0);
    dbLayout->addWidget(dbPathEdit_);
    auto* dbPickBtn = new QPushButton("Browse...", dbRow);
    dbLayout->addWidget(dbPickBtn);
    formLayout->addRow("SQLite DB:", dbRow);

    queryPathEdit_ = new QLineEdit(inputGroup);
    auto* queryRow = new QWidget(inputGroup);
    auto* queryLayout = new QHBoxLayout(queryRow);
    queryLayout->setContentsMargins(0, 0, 0, 0);
    queryLayout->addWidget(queryPathEdit_);
    auto* queryPickBtn = new QPushButton("Browse...", queryRow);
    queryLayout->addWidget(queryPickBtn);
    formLayout->addRow("Query image:", queryRow);

    topKEdit_ = new QLineEdit(inputGroup);
    topKEdit_->setText("5");
    formLayout->addRow("Top-K:", topKEdit_);

    auto* actionRow = new QWidget(inputGroup);
    auto* actionLayout = new QHBoxLayout(actionRow);
    actionLayout->setContentsMargins(0, 0, 0, 0);
    indexButton_ = new QPushButton("Index Dataset", actionRow);
    searchButton_ = new QPushButton("Search Top-K", actionRow);
    actionLayout->addWidget(indexButton_);
    actionLayout->addWidget(searchButton_);
    formLayout->addRow("Actions:", actionRow);

    auto* previewGroup = new QGroupBox("Visualization", central);
    auto* previewLayout = new QVBoxLayout(previewGroup);
    auto* queryTitle = new QLabel("Input image:", previewGroup);
    queryPreviewLabel_ = new QLabel(previewGroup);
    queryPreviewLabel_->setMinimumSize(300, 220);
    queryPreviewLabel_->setAlignment(Qt::AlignCenter);
    queryPreviewLabel_->setStyleSheet("border: 1px solid #999;");
    queryPreviewLabel_->setText("No query image");
    previewLayout->addWidget(queryTitle);
    previewLayout->addWidget(queryPreviewLabel_);

    auto* pipelineTitle = new QLabel("Input processing pipeline:", previewGroup);
    previewLayout->addWidget(pipelineTitle);
    auto* pipelineLayout = new QHBoxLayout();
    for (int i = 0; i < 5; ++i) {
        auto* card = new QWidget(previewGroup);
        auto* cardLayout = new QVBoxLayout(card);
        cardLayout->setContentsMargins(4, 4, 4, 4);
        auto* imageLabel = new QLabel(card);
        imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setMinimumSize(130, 96);
        imageLabel->setStyleSheet("border: 1px solid #999;");
        imageLabel->setText("Step");
        imageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        auto* textLabel = new QLabel(card);
        textLabel->setAlignment(Qt::AlignCenter);
        textLabel->setWordWrap(true);
        textLabel->setText("-");
        cardLayout->addWidget(imageLabel);
        cardLayout->addWidget(textLabel);
        pipelineLayout->addWidget(card, 1);
        pipelineImageLabels_[i] = imageLabel;
        pipelineTextLabels_[i] = textLabel;
    }
    previewLayout->addLayout(pipelineLayout);

    auto* top5Title = new QLabel("Top-5 matched birds:", previewGroup);
    previewLayout->addWidget(top5Title);
    auto* top5Layout = new QHBoxLayout();
    for (int i = 0; i < 5; ++i) {
        auto* card = new QWidget(previewGroup);
        auto* cardLayout = new QVBoxLayout(card);
        cardLayout->setContentsMargins(4, 4, 4, 4);
        auto* imageLabel = new QLabel(card);
        imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setMinimumSize(150, 110);
        imageLabel->setStyleSheet("border: 1px solid #999;");
        imageLabel->setText(QString("Top %1").arg(i + 1));
        imageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        auto* infoLabel = new QLabel(card);
        infoLabel->setAlignment(Qt::AlignCenter);
        infoLabel->setWordWrap(true);
        infoLabel->setText("-");
        cardLayout->addWidget(imageLabel);
        cardLayout->addWidget(infoLabel);
        top5Layout->addWidget(card, 1);
        topImageLabels_[i] = imageLabel;
        topInfoLabels_[i] = infoLabel;
    }
    previewLayout->addLayout(top5Layout);

    resultTable_ = new QTableWidget(0, 4, central);
    resultTable_->setHorizontalHeaderLabels({"Rank", "Distance", "Label", "Image Path"});
    resultTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    resultTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    resultTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    resultTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    resultTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    resultTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

    auto* logLabel = new QLabel("Logs:", central);
    logView_ = new QTextEdit(central);
    logView_->setReadOnly(true);

    rootLayout->addWidget(inputGroup);
    rootLayout->addWidget(previewGroup);
    rootLayout->addWidget(new QLabel("Search results:", central));
    rootLayout->addWidget(resultTable_, 1);
    rootLayout->addWidget(logLabel);
    rootLayout->addWidget(logView_, 1);

    setCentralWidget(central);

    connect(datasetPickBtn, &QPushButton::clicked, this, &MainWindow::pickDatasetFolder);
    connect(dbPickBtn, &QPushButton::clicked, this, &MainWindow::pickDbFile);
    connect(queryPickBtn, &QPushButton::clicked, this, &MainWindow::pickQueryImage);
    connect(indexButton_, &QPushButton::clicked, this, &MainWindow::runIndexing);
    connect(searchButton_, &QPushButton::clicked, this, &MainWindow::runSearch);
}

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

void MainWindow::runIndexing() {
    Config cfg;
    cfg.dataset_path = datasetPathEdit_->text().toStdString();
    cfg.db_path = dbPathEdit_->text().toStdString();

    ImagePreprocessor preprocessor(cfg.resize_width, cfg.resize_height);
    FeatureExtractor extractor(8);
    SqliteRepo repo(cfg.db_path);

    if (!repo.open() || !repo.initSchema()) {
        QMessageBox::critical(this, "Error", "Cannot open/init SQLite database.");
        return;
    }

    appendLog("Start indexing: " + QString::fromStdString(cfg.dataset_path));
    setEnabled(false);

    Indexer indexer(preprocessor, extractor, repo);
    const bool ok = indexer.run(cfg.dataset_path);

    setEnabled(true);
    if (ok) {
        appendLog("Indexing completed.");
        QMessageBox::information(this, "Done", "Indexing completed successfully.");
    } else {
        appendLog("Indexing failed.");
        QMessageBox::warning(this, "Failed", "Indexing failed. See logs for details.");
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
    updatePipelineVisualization(QString::fromStdString(cfg.query_image_path));

    ImagePreprocessor preprocessor(cfg.resize_width, cfg.resize_height);
    FeatureExtractor extractor(8);
    SqliteRepo repo(cfg.db_path);
    if (!repo.open() || !repo.initSchema()) {
        QMessageBox::critical(this, "Error", "Cannot open/init SQLite database.");
        return;
    }

    Searcher searcher(preprocessor, extractor, repo, cfg);
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
    target->setPixmap(pix.scaled(width, height, Qt::KeepAspectRatio, Qt::SmoothTransformation));
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

    cv::Mat lab;
    cv::cvtColor(resized, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);
    cv::Mat lChannelBgr;
    cv::cvtColor(labChannels[0], lChannelBgr, cv::COLOR_GRAY2BGR);

    cv::Mat mask = buildForegroundMaskForUi(resized);
    cv::Mat maskBgr;
    cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);

    cv::Mat foregroundOnly;
    cv::bitwise_and(resized, resized, foregroundOnly, mask);

    cv::Mat weightMap(resized.size(), CV_32F, cv::Scalar(0));
    for (int r = 0; r < resized.rows; ++r) {
        float* wptr = weightMap.ptr<float>(r);
        const uint8_t* mptr = mask.ptr<uint8_t>(r);
        for (int c = 0; c < resized.cols; ++c) {
            float w = 0.25f + centerWeight(r, c, resized.rows, resized.cols);
            if (mptr[c] > 0) {
                w *= 1.8f;
            }
            wptr[c] = w;
        }
    }
    cv::Mat weightNorm, weightColor;
    cv::normalize(weightMap, weightNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(weightNorm, weightColor, cv::COLORMAP_JET);

    const std::array<cv::Mat, 5> stepImages = {resized, lChannelBgr, maskBgr, foregroundOnly, weightColor};
    const std::array<QString, 5> stepNames = {QString("1) Resize 256x256"), QString("2) Lab-L channel"),
                                              QString("3) Foreground mask"), QString("4) Foreground only"),
                                              QString("5) PWH weight map")};

    for (int i = 0; i < 5; ++i) {
        pipelineImageLabels_[i]->setText("");
        pipelineImageLabels_[i]->setPixmap(matToPixmap(stepImages[i]).scaled(
            130, 96, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        pipelineTextLabels_[i]->setText(stepNames[i]);
    }
}

}  // namespace cbir
