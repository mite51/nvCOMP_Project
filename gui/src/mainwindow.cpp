/**
 * @file mainwindow.cpp
 * @brief Implementation of MainWindow class
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "compression_worker.h"
#include "archive_viewer.h"
#include <QMessageBox>
#include <QApplication>
#include <QFileDialog>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QDir>
#include <QFileInfo>
#include <QUrl>
#include <QPushButton>
#include <QListView>
#include <QTreeView>
#include <QAbstractItemView>
#include <QEvent>
#include <QListWidgetItem>

// Helper class to keep the file dialog button always enabled
class ButtonEnabledFilter : public QObject
{
public:
    ButtonEnabledFilter(QPushButton* button, QObject* parent = nullptr)
        : QObject(parent), m_button(button) {}
    
protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (watched == m_button && event->type() == QEvent::EnabledChange) {
            if (!m_button->isEnabled()) {
                m_button->setEnabled(true);
            }
        }
        return QObject::eventFilter(watched, event);
    }
    
private:
    QPushButton* m_button;
};

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_initialized(false)
    , m_gpuAvailable(false)
    , m_worker(nullptr)
{
    ui->setupUi(this);
    setupUi();
    setupConnections();
    checkGpuAvailability();
    updateUiState();
    m_initialized = true;
}

MainWindow::~MainWindow()
{
    // Clean up worker if it exists
    if (m_worker) {
        if (m_worker->isRunning()) {
            m_worker->cancel();
            // Wait up to 5 seconds for thread to finish
            if (!m_worker->wait(5000)) {
                m_worker->terminate();
                m_worker->wait();  // Wait for termination to complete
            }
        }
        delete m_worker;
        m_worker = nullptr;
    }
    
    delete ui;
}

void MainWindow::setupUi()
{
    // Set window properties
    setWindowTitle("nvCOMP - GPU Accelerated Compression");
    resize(900, 700);
    
    // Set minimum size to ensure usability
    setMinimumSize(640, 480);
    
    // Enable drag and drop
    setAcceptDrops(true);
    
    // Center window on screen (will be enhanced in future tasks)
    // For now, let Qt position it
}

void MainWindow::setupConnections()
{
    // Connect menu actions
    if (ui->actionAbout) {
        connect(ui->actionAbout, &QAction::triggered, 
                this, &MainWindow::onAboutTriggered);
    }
    
    if (ui->actionExit) {
        connect(ui->actionExit, &QAction::triggered, 
                this, &MainWindow::onExitTriggered);
    }
    
    if (ui->actionAddFiles) {
        connect(ui->actionAddFiles, &QAction::triggered,
                this, &MainWindow::onAddFilesClicked);
    }
    
    if (ui->actionViewArchive) {
        connect(ui->actionViewArchive, &QAction::triggered,
                this, &MainWindow::onViewArchiveTriggered);
    }
    
    // Connect file list double-click
    connect(ui->listWidgetFiles, &QListWidget::itemDoubleClicked,
            this, &MainWindow::onFileListDoubleClicked);
    
    // Connect file buttons
    connect(ui->buttonAddFiles, &QPushButton::clicked,
            this, &MainWindow::onAddFilesClicked);
    connect(ui->buttonClearSelected, &QPushButton::clicked,
            this, &MainWindow::onClearSelectedClicked);
    connect(ui->buttonClearFiles, &QPushButton::clicked,
            this, &MainWindow::onClearFilesClicked);
    
    // Connect action buttons
    connect(ui->buttonCompress, &QPushButton::clicked,
            this, &MainWindow::onCompressClicked);
    // buttonDecompress is hidden - this is an archive creation window
    // connect(ui->buttonDecompress, &QPushButton::clicked,
    //         this, &MainWindow::onDecompressClicked);
    connect(ui->buttonSettings, &QPushButton::clicked,
            this, &MainWindow::onSettingsClicked);
    connect(ui->buttonBrowseOutput, &QPushButton::clicked,
            this, &MainWindow::onBrowseOutputClicked);
    
    // Connect algorithm selection
    connect(ui->comboBoxAlgorithm, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onAlgorithmChanged);
    
    // Connect CPU mode checkbox
    connect(ui->checkBoxCpuMode, &QCheckBox::toggled,
            this, &MainWindow::onCpuModeToggled);
}

void MainWindow::checkGpuAvailability()
{
    // Check for CUDA GPU availability using nvCOMP API
    m_gpuAvailable = nvcomp_is_cuda_available();
    
    if (m_gpuAvailable) {
        ui->labelGpuStatus->setText("GPU Status: ✅ GPU Available");
        ui->labelGpuIcon->setText("🎮");
        ui->checkBoxCpuMode->setChecked(false);
    } else {
        ui->labelGpuStatus->setText("GPU Status: ⚠️ No GPU detected - CPU mode only");
        ui->labelGpuIcon->setText("💻");
        ui->checkBoxCpuMode->setChecked(true);
        ui->checkBoxCpuMode->setEnabled(false);
    }
}

void MainWindow::updateUiState()
{
    // Enable/disable compress button based on file list
    bool hasFiles = !m_fileList.isEmpty();
    ui->buttonCompress->setEnabled(hasFiles);
    // buttonDecompress is hidden - this is an archive creation window
    // ui->buttonDecompress->setEnabled(hasFiles);
    
    // Update status label
    if (hasFiles) {
        ui->labelStatus->setText(QString("Ready - %1 file(s) selected").arg(m_fileList.count()));
    } else {
        ui->labelStatus->setText("Ready - No files selected");
    }
}

void MainWindow::addFiles(const QStringList &paths)
{
    for (const QString &path : paths) {
        QFileInfo fileInfo(path);
        
        // Skip if already in list
        if (m_fileList.contains(path)) {
            continue;
        }
        
        // Add both files and folders directly to the list
        // The core library will handle recursive folder traversal
        if (fileInfo.exists()) {
            m_fileList.append(path);
            
            // Show with appropriate icon/indicator
            QString displayText = fileInfo.absoluteFilePath();
            if (fileInfo.isDir()) {
                displayText += "/";  // Visual indicator for folders
            }
            ui->listWidgetFiles->addItem(displayText);
        }
    }
    
    updateUiState();
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    // Accept drag if it contains URLs (files)
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent *event)
{
    const QMimeData *mimeData = event->mimeData();
    
    if (mimeData->hasUrls()) {
        QStringList pathsToAdd;
        
        for (const QUrl &url : mimeData->urls()) {
            QString path = url.toLocalFile();
            QFileInfo fileInfo(path);
            
            if (fileInfo.exists()) {
                // Add both files and folders directly
                // The core library will handle recursive folder traversal
                pathsToAdd.append(path);
            }
        }
        
        if (!pathsToAdd.isEmpty()) {
            addFiles(pathsToAdd);
        }
        
        event->acceptProposedAction();
    }
}

QString MainWindow::windowTitle() const
{
    return QMainWindow::windowTitle();
}

bool MainWindow::isInitialized() const
{
    return m_initialized;
}

QStringList MainWindow::getFileList() const
{
    return m_fileList;
}

QString MainWindow::getSelectedAlgorithm() const
{
    // Get the user data which contains the actual algorithm name
    QString algoData = ui->comboBoxAlgorithm->currentData().toString();
    if (!algoData.isEmpty()) {
        return algoData;
    }
    
    // Fallback: parse algorithm name from display text
    // Display format: "AlgoName [GPU/CPU] - Description"
    QString displayText = ui->comboBoxAlgorithm->currentText();
    
    // Extract just the algorithm name (first word)
    int spaceIdx = displayText.indexOf(' ');
    if (spaceIdx > 0) {
        return displayText.left(spaceIdx);
    }
    
    return displayText;
}

bool MainWindow::isCpuModeEnabled() const
{
    return ui->checkBoxCpuMode->isChecked();
}

bool MainWindow::isVolumesEnabled() const
{
    return ui->checkBoxVolumes->isChecked();
}

int MainWindow::getVolumeSize() const
{
    return ui->spinBoxVolumeSize->value();
}

QString MainWindow::getOutputArchiveName() const
{
    return ui->lineEditOutputArchive->text();
}

void MainWindow::onAboutTriggered()
{
    QMessageBox::about(this, 
        tr("About nvCOMP"),
        tr("<h3>nvCOMP GUI v1.0.0</h3>"
           "<p>GPU-accelerated compression tool using NVIDIA nvCOMP</p>"
           "<p><b>Supported Algorithms:</b></p>"
           "<ul>"
           "<li>LZ4 - Fast compression</li>"
           "<li>Snappy - Very fast compression</li>"
           "<li>Zstd - Best compression ratios</li>"
           "<li>GDeflate - GPU-optimized DEFLATE</li>"
           "<li>ANS - Asymmetric Numeral Systems</li>"
           "<li>Bitcomp - Lossless numerical compression</li>"
           "</ul>"
           "<p>Copyright © 2024</p>"));
}

void MainWindow::onExitTriggered()
{
    // Clean exit (future tasks will add cleanup for workers/threads)
    QApplication::quit();
}

void MainWindow::onAddFilesClicked()
{
    // Use custom QFileDialog to allow selecting both files AND folders simultaneously
    // Based on: https://www.qtcentre.org/threads/43841-QFileDialog-to-select-files-AND-folders
    QFileDialog* fileDialog = new QFileDialog(this);
    fileDialog->setFileMode(QFileDialog::Directory);
    fileDialog->setOption(QFileDialog::DontUseNativeDialog, true);
    fileDialog->setWindowTitle(tr("Select Files and/or Folders"));
    
    // Enable multi-selection for both list and tree views
    QListView *listView = fileDialog->findChild<QListView*>("listView");
    if (listView) {
        listView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    }
    
    QTreeView *treeView = fileDialog->findChild<QTreeView*>();
    if (treeView) {
        treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    }
    
    // Find and keep the "Open"/"Choose" button always enabled
    // This fixes the issue where Ctrl+clicking files disables the button
    QPushButton *openButton = nullptr;
    QList<QPushButton*> buttons = fileDialog->findChildren<QPushButton*>();
    for (QPushButton* btn : buttons) {
        QString text = btn->text().toLower();
        if (text.contains("open") || text.contains("choose")) {
            openButton = btn;
            break;
        }
    }
    
    if (openButton) {
        // Install event filter to keep button always enabled
        // This ensures the button stays active even when selecting files
        ButtonEnabledFilter* filter = new ButtonEnabledFilter(openButton, fileDialog);
        openButton->installEventFilter(filter);
        openButton->setEnabled(true);
        
        // Override the button behavior - accept the dialog using public methods
        openButton->disconnect();
        QObject::connect(openButton, &QPushButton::clicked, fileDialog, [fileDialog]() {
            fileDialog->setResult(QDialog::Accepted);
            fileDialog->hide();
        });
    }
    
    if (fileDialog->exec() == QDialog::Accepted) {
        QStringList selectedPaths;
        
        // Get selected items from the list view
        if (listView && listView->selectionModel()) {
            QModelIndexList indexes = listView->selectionModel()->selectedIndexes();
            QDir currentDir = fileDialog->directory();
            
            for (const QModelIndex &index : indexes) {
                if (index.column() == 0) {
                    QString fileName = index.data().toString();
                    QString fullPath = currentDir.absoluteFilePath(fileName);
                    if (!selectedPaths.contains(fullPath)) {
                        selectedPaths.append(fullPath);
                    }
                }
            }
        }
        
        // Fallback to selectedFiles() if nothing was found in list view
        if (selectedPaths.isEmpty()) {
            selectedPaths = fileDialog->selectedFiles();
        }
        
        if (!selectedPaths.isEmpty()) {
            addFiles(selectedPaths);
        }
    }
    
    delete fileDialog;
}

void MainWindow::onAddFolderClicked()
{
    // Deprecated - forward to onAddFilesClicked which now handles both
    onAddFilesClicked();
}

void MainWindow::onClearSelectedClicked()
{
    // Get selected items from the list widget
    QList<QListWidgetItem*> selectedItems = ui->listWidgetFiles->selectedItems();
    
    if (selectedItems.isEmpty()) {
        return; // Nothing selected
    }
    
    // Remove selected items from both the UI and internal list
    for (QListWidgetItem* item : selectedItems) {
        QString filePath = item->text();
        
        // Remove from internal list
        m_fileList.removeAll(filePath);
        
        // Remove from UI (this deletes the item)
        delete ui->listWidgetFiles->takeItem(ui->listWidgetFiles->row(item));
    }
    
    updateUiState();
}

void MainWindow::onClearFilesClicked()
{
    m_fileList.clear();
    ui->listWidgetFiles->clear();
    ui->progressBar->setValue(0);
    updateUiState();
}

void MainWindow::onCompressClicked()
{
    // Check if this is a cancel request
    if (m_worker && m_worker->isRunning()) {
        // User clicked Cancel - stop the operation
        m_worker->cancel();
        
        // Disable button while waiting for worker to stop
        ui->buttonCompress->setEnabled(false);
        ui->buttonCompress->setText("⏳ Stopping...");
        ui->buttonCompress->setToolTip("Waiting for operation to stop");
        statusBar()->showMessage("Cancelling operation...");
        
        // Wait for worker to actually stop (up to 3 seconds)
        if (!m_worker->wait(3000)) {
            m_worker->terminate();
            m_worker->wait();
        }
        
        // Restore button
        ui->buttonCompress->setText("🗜️ Compress");
        ui->buttonCompress->setToolTip("Compress selected files");
        ui->buttonCompress->setEnabled(true);
        statusBar()->showMessage("Operation cancelled", 3000);
        ui->progressBar->setValue(0);
        
        return;
    }
    
    if (m_fileList.isEmpty()) {
        QMessageBox::warning(this, tr("No Files"), tr("Please add files to compress first."));
        return;
    }
    
    // Get settings
    QString algorithm = getSelectedAlgorithm();
    QString outputPath = getOutputArchiveName();
    bool useCpuMode = isCpuModeEnabled();
    uint64_t volumeSize = 0;
    
    if (isVolumesEnabled()) {
        volumeSize = static_cast<uint64_t>(getVolumeSize()) * 1024 * 1024;  // Convert MB to bytes
    }
    
    // Create worker if it doesn't exist
    if (!m_worker) {
        m_worker = new CompressionWorker(this);
        
        // Connect signals
        connect(m_worker, &CompressionWorker::progressChanged,
                this, &MainWindow::onWorkerProgress);
        connect(m_worker, &CompressionWorker::progressDetails,
                this, &MainWindow::onWorkerProgressDetails);
        connect(m_worker, &CompressionWorker::finished,
                this, &MainWindow::onWorkerFinished);
        connect(m_worker, &CompressionWorker::error,
                this, &MainWindow::onWorkerError);
        connect(m_worker, &CompressionWorker::canceled,
                this, &MainWindow::onWorkerCanceled);
        connect(m_worker, &CompressionWorker::statusMessage,
                this, &MainWindow::onWorkerStatusMessage);
    }
    
    // Setup and start compression
    m_worker->setupCompress(m_fileList, outputPath, algorithm, useCpuMode, volumeSize);
    m_worker->start();
    
    // Update UI state - Change Compress button to Cancel
    ui->buttonCompress->setText("❌ Cancel");
    ui->buttonCompress->setToolTip("Cancel compression operation");
    ui->buttonCompress->setEnabled(true);  // Keep enabled so user can cancel
    // ui->buttonDecompress->setEnabled(false);  // Hidden - archive creation window
    ui->buttonClearFiles->setEnabled(false);
    ui->buttonClearSelected->setEnabled(false);
    ui->progressBar->setValue(0);
    statusBar()->showMessage("Compression started...");
}

void MainWindow::onDecompressClicked()
{
    if (m_fileList.isEmpty()) {
        QMessageBox::warning(this, tr("No Files"), tr("Please add files to decompress first."));
        return;
    }
    
    // Check if worker is already running
    if (m_worker && m_worker->isRunning()) {
        QMessageBox::warning(this, tr("Operation in Progress"), 
                           tr("An operation is already in progress. Please wait or cancel it."));
        return;
    }
    
    // Get output directory
    QString outputPath = QFileDialog::getExistingDirectory(
        this,
        tr("Select Output Directory"),
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (outputPath.isEmpty()) {
        return;  // User canceled
    }
    
    // Get settings
    bool useCpuMode = isCpuModeEnabled();
    
    // Create worker if it doesn't exist
    if (!m_worker) {
        m_worker = new CompressionWorker(this);
        
        // Connect signals
        connect(m_worker, &CompressionWorker::progressChanged,
                this, &MainWindow::onWorkerProgress);
        connect(m_worker, &CompressionWorker::progressDetails,
                this, &MainWindow::onWorkerProgressDetails);
        connect(m_worker, &CompressionWorker::finished,
                this, &MainWindow::onWorkerFinished);
        connect(m_worker, &CompressionWorker::error,
                this, &MainWindow::onWorkerError);
        connect(m_worker, &CompressionWorker::canceled,
                this, &MainWindow::onWorkerCanceled);
        connect(m_worker, &CompressionWorker::statusMessage,
                this, &MainWindow::onWorkerStatusMessage);
    }
    
    // Setup and start decompression
    m_worker->setupDecompress(m_fileList, outputPath, QString(), useCpuMode);
    m_worker->start();
    
    // Update UI state
    ui->buttonCompress->setEnabled(false);
    // ui->buttonDecompress->setEnabled(false);  // Hidden - archive creation window
    ui->buttonClearFiles->setEnabled(false);
    ui->buttonClearSelected->setEnabled(false);
    ui->progressBar->setValue(0);
    statusBar()->showMessage("Decompression started...");
}

void MainWindow::onSettingsClicked()
{
    // Stub for now - will be implemented in future tasks
    QMessageBox::information(this,
        tr("Settings"),
        tr("Settings dialog will be implemented in Task 2.4: Settings and Configuration")
    );
}

void MainWindow::onAlgorithmChanged(int index)
{
    // Update status when algorithm changes
    QString algorithm = ui->comboBoxAlgorithm->itemText(index);
    statusBar()->showMessage(QString("Algorithm changed to: %1").arg(algorithm), 3000);
}

void MainWindow::onCpuModeToggled(bool checked)
{
    // Update status when CPU mode is toggled
    if (checked) {
        statusBar()->showMessage("CPU mode enabled", 3000);
    } else {
        statusBar()->showMessage("GPU mode enabled", 3000);
    }
}

void MainWindow::onBrowseOutputClicked()
{
    // Open file dialog to select output archive location and name
    QString fileName = QFileDialog::getSaveFileName(
        this,
        tr("Select Output Archive Name"),
        ui->lineEditOutputArchive->text().isEmpty() ? "archive.nvcomp" : ui->lineEditOutputArchive->text(),
        tr("nvCOMP Archives (*.nvcomp);;All Files (*.*)")
    );
    
    if (!fileName.isEmpty()) {
        ui->lineEditOutputArchive->setText(fileName);
    }
}

void MainWindow::onWorkerProgress(int percentage, const QString &currentFile)
{
    ui->progressBar->setValue(percentage);
    
    if (!currentFile.isEmpty()) {
        statusBar()->showMessage(QString("Processing: %1 (%2%)").arg(currentFile).arg(percentage));
    } else {
        statusBar()->showMessage(QString("Progress: %1%").arg(percentage));
    }
}

void MainWindow::onWorkerProgressDetails(uint64_t current, uint64_t total, double speedMBps, int etaSeconds)
{
    // Format size in human-readable form
    auto formatSize = [](uint64_t bytes) -> QString {
        if (bytes < 1024) return QString("%1 B").arg(bytes);
        if (bytes < 1024 * 1024) return QString("%1 KB").arg(bytes / 1024.0, 0, 'f', 2);
        if (bytes < 1024 * 1024 * 1024) return QString("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 2);
        return QString("%1 GB").arg(bytes / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2);
    };
    
    // Format ETA
    auto formatTime = [](int seconds) -> QString {
        if (seconds < 60) return QString("%1s").arg(seconds);
        if (seconds < 3600) return QString("%1m %2s").arg(seconds / 60).arg(seconds % 60);
        return QString("%1h %2m").arg(seconds / 3600).arg((seconds % 3600) / 60);
    };
    
    QString detailsText = QString("%1 / %2 @ %3 MB/s (ETA: %4)")
                             .arg(formatSize(current))
                             .arg(formatSize(total))
                             .arg(speedMBps, 0, 'f', 2)
                             .arg(formatTime(etaSeconds));
    
    ui->labelStatus->setText(detailsText);
}

void MainWindow::onWorkerFinished(const QString &outputPath, double compressionRatio, qint64 durationMs)
{
    // Re-enable buttons and restore Compress button text
    ui->buttonCompress->setText("🗜️ Compress");
    ui->buttonCompress->setToolTip("Compress selected files");
    ui->buttonCompress->setEnabled(true);
    // ui->buttonDecompress->setEnabled(true);  // Hidden - archive creation window
    ui->buttonClearFiles->setEnabled(true);
    ui->buttonClearSelected->setEnabled(true);
    
    // Update progress bar to 100%
    ui->progressBar->setValue(100);
    
    // Format duration
    double durationSec = durationMs / 1000.0;
    QString durationText;
    if (durationSec < 60) {
        durationText = QString("%1s").arg(durationSec, 0, 'f', 2);
    } else {
        int minutes = static_cast<int>(durationSec / 60);
        double seconds = durationSec - (minutes * 60);
        durationText = QString("%1m %2s").arg(minutes).arg(seconds, 0, 'f', 1);
    }
    
    // Show completion message
    QString message;
    if (compressionRatio > 0.0) {
        // Compression operation
        double compressionPercent = compressionRatio * 100.0;
        message = tr("Compression completed successfully!\n\n"
                    "Output: %1\n"
                    "Compression ratio: %2% (smaller is better)\n"
                    "Time taken: %3")
                    .arg(outputPath)
                    .arg(compressionPercent, 0, 'f', 2)
                    .arg(durationText);
    } else {
        // Decompression operation
        message = tr("Decompression completed successfully!\n\n"
                    "Output: %1\n"
                    "Time taken: %2")
                    .arg(outputPath)
                    .arg(durationText);
    }
    
    QMessageBox::information(this, tr("Success"), message);
    
    statusBar()->showMessage(tr("Operation completed in %1").arg(durationText), 5000);
}

void MainWindow::onWorkerError(const QString &errorMessage)
{
    // Re-enable buttons and restore Compress button text
    ui->buttonCompress->setText("🗜️ Compress");
    ui->buttonCompress->setToolTip("Compress selected files");
    ui->buttonCompress->setEnabled(true);
    // ui->buttonDecompress->setEnabled(true);  // Hidden - archive creation window
    ui->buttonClearFiles->setEnabled(true);
    ui->buttonClearSelected->setEnabled(true);
    
    // Show error message
    QMessageBox::critical(this, tr("Error"), 
                         tr("Operation failed:\n\n%1").arg(errorMessage));
    
    statusBar()->showMessage(tr("Operation failed"), 5000);
    ui->progressBar->setValue(0);
}

void MainWindow::onWorkerCanceled()
{
    // Re-enable buttons and restore Compress button text
    ui->buttonCompress->setText("🗜️ Compress");
    ui->buttonCompress->setToolTip("Compress selected files");
    ui->buttonCompress->setEnabled(true);
    // ui->buttonDecompress->setEnabled(true);  // Hidden - archive creation window
    ui->buttonClearFiles->setEnabled(true);
    ui->buttonClearSelected->setEnabled(true);
    
    QMessageBox::information(this, tr("Canceled"), tr("Operation was canceled."));
    
    statusBar()->showMessage(tr("Operation canceled"), 3000);
    ui->progressBar->setValue(0);
}

void MainWindow::onWorkerStatusMessage(const QString &message)
{
    statusBar()->showMessage(message, 3000);
}

void MainWindow::onViewArchiveTriggered()
{
    // Open file dialog to select archive
    QString archivePath = QFileDialog::getOpenFileName(
        this,
        tr("Select Archive to View"),
        QString(),
        tr("nvCOMP Archives (*.nvcomp *.nvcomp.*);;All Files (*.*)")
    );
    
    if (archivePath.isEmpty()) {
        return;  // User canceled
    }
    
    // Check if file exists
    if (!QFile::exists(archivePath)) {
        QMessageBox::warning(this, tr("File Not Found"),
            tr("The selected archive file does not exist:\n%1").arg(archivePath));
        return;
    }
    
    // Open archive viewer dialog
    ArchiveViewerDialog dialog(archivePath, this);
    dialog.exec();
}

void MainWindow::onFileListDoubleClicked(QListWidgetItem* item)
{
    if (!item) {
        return;
    }
    
    QString filePath = item->text();
    QFileInfo fileInfo(filePath);
    
    // Check if this is a compressed archive file
    QString suffix = fileInfo.suffix().toLower();
    QString fileName = fileInfo.fileName().toLower();
    
    // Check for archive extensions
    bool isArchive = suffix == "nvcomp" || 
                     fileName.contains(".nvcomp.") ||
                     fileName.endsWith(".nvcomp");
    
    if (isArchive) {
        // Open archive viewer
        ArchiveViewerDialog dialog(filePath, this);
        dialog.exec();
    } else {
        // For non-archive files, show info message
        QMessageBox::information(this, tr("File Info"),
            tr("File: %1\n\nDouble-click archive files (*.nvcomp) to view their contents.")
                .arg(fileInfo.fileName()));
    }
}


