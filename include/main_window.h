#pragma once

#include <QMainWindow>

#include <array>
#include <memory>
#include <string>

#include "sqlite_repo.h"

QT_BEGIN_NAMESPACE
class QLineEdit;
class QLabel;
class QPushButton;
class QTextEdit;
class QTableWidget;
QT_END_NAMESPACE

namespace cbir {
class MainWindow : public QMainWindow {
    Q_OBJECT

   public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

   private slots:
    void pickDatasetFolder();
    void pickDbFile();
    void pickQueryImage();
    void runSearch();
    void loadIndexToRam();

   private:
    QLineEdit* datasetPathEdit_ = nullptr;
    QLineEdit* dbPathEdit_ = nullptr;
    QLineEdit* queryPathEdit_ = nullptr;
    QLineEdit* topKEdit_ = nullptr;
    QPushButton* searchButton_ = nullptr;

    void runIndexing(bool silent = false);
    QTextEdit* logView_ = nullptr;
    QTableWidget* resultTable_ = nullptr;
    QLabel* queryPreviewLabel_ = nullptr;
    std::array<QLabel*, 5> pipelineImageLabels_{};
    std::array<QLabel*, 5> pipelineTextLabels_{};
    std::array<QLabel*, 5> topImageLabels_{};
    std::array<QLabel*, 5> topInfoLabels_{};

    std::unique_ptr<SqliteRepo> persistentRepo_;
    std::string loadedDbPath_;

    void appendLog(const QString& message);
    void setPreviewImage(QLabel* target, const QString& imagePath, int width, int height);
    void updatePipelineVisualization(const QString& imagePath);
};
}  // namespace cbir
