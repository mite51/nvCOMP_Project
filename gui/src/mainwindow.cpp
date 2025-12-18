/**
 * @file mainwindow.cpp
 * @brief Implementation of MainWindow class
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "compression_worker.h"
#include "archive_viewer.h"
#include "settings_dialog.h"
#include "gpu_monitor.h"
#include "progress_widget.h"
#include <QMessageBox>
#include <QApplication>
#include <QPalette>
#include <QStyleHints>
#include <QColor>
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
#include <QDebug>

#ifdef _WIN32
#include "../../platform/windows/context_menu.h"
#include "../../platform/windows/file_associations.h"
#endif

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
    , m_settingsDialog(nullptr)
    , m_gpuMonitor(nullptr)
    , m_progressWidget(nullptr)
    , m_currentStage("")
{
    ui->setupUi(this);
    setupUi();
    setupConnections();
    checkGpuAvailability();
    updateUiState();
    applySettingsToUi();  // Load and apply saved settings
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
    
    // Clean up settings dialog if it exists
    if (m_settingsDialog) {
        delete m_settingsDialog;
        m_settingsDialog = nullptr;
    }
    
    // Clean up GPU monitor if it exists
    if (m_gpuMonitor) {
        delete m_gpuMonitor;
        m_gpuMonitor = nullptr;
    }
    
    // Clean up progress widget if it exists
    if (m_progressWidget) {
        delete m_progressWidget;
        m_progressWidget = nullptr;
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
    
    // Manually set userData for algorithm combo box (Qt Designer userData sometimes doesn't work)
    ui->comboBoxAlgorithm->setItemData(0, "LZ4");
    ui->comboBoxAlgorithm->setItemData(1, "Snappy");
    ui->comboBoxAlgorithm->setItemData(2, "Zstd");
    ui->comboBoxAlgorithm->setItemData(3, "GDeflate");
    ui->comboBoxAlgorithm->setItemData(4, "ANS");
    ui->comboBoxAlgorithm->setItemData(5, "Bitcomp");
    
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
    
    if (ui->actionSettings) {
        connect(ui->actionSettings, &QAction::triggered,
                this, &MainWindow::onSettingsClicked);
    }
    
    // Check if GPU Monitor action exists (may need to be added to UI file)
    if (ui->menuTools) {
        // Try to find existing GPU Monitor action
        QAction *gpuMonitorAction = nullptr;
        for (QAction *action : ui->menuTools->actions()) {
            if (action->text().contains("GPU Monitor", Qt::CaseInsensitive)) {
                gpuMonitorAction = action;
                break;
            }
        }
        
        // If not found, create it
        if (!gpuMonitorAction) {
            gpuMonitorAction = new QAction("GPU Monitor", this);
            ui->menuTools->addAction(gpuMonitorAction);
        }
        
        connect(gpuMonitorAction, &QAction::triggered,
                this, &MainWindow::onGPUMonitorTriggered);
        
#ifdef _WIN32
        // Add context menu registration options (Windows only)
        ui->menuTools->addSeparator();
        
        QAction *registerContextMenuAction = new QAction("Register Windows Context Menu...", this);
        registerContextMenuAction->setToolTip("Add nvCOMP to Windows Explorer right-click menu (requires admin)");
        ui->menuTools->addAction(registerContextMenuAction);
        connect(registerContextMenuAction, &QAction::triggered,
                this, &MainWindow::onRegisterContextMenu);
        
        QAction *unregisterContextMenuAction = new QAction("Unregister Windows Context Menu...", this);
        unregisterContextMenuAction->setToolTip("Remove nvCOMP from Windows Explorer right-click menu (requires admin)");
        ui->menuTools->addAction(unregisterContextMenuAction);
        connect(unregisterContextMenuAction, &QAction::triggered,
                this, &MainWindow::onUnregisterContextMenu);
        
        // Add file association registration options (Windows only)
        ui->menuTools->addSeparator();
        
        QAction *registerFileAssocAction = new QAction("Register File Associations...", this);
        registerFileAssocAction->setToolTip("Associate compressed file types with nvCOMP (requires admin)");
        ui->menuTools->addAction(registerFileAssocAction);
        connect(registerFileAssocAction, &QAction::triggered,
                this, &MainWindow::onRegisterFileAssociations);
        
        QAction *unregisterFileAssocAction = new QAction("Unregister File Associations...", this);
        unregisterFileAssocAction->setToolTip("Remove file associations for compressed types (requires admin)");
        ui->menuTools->addAction(unregisterFileAssocAction);
        connect(unregisterFileAssocAction, &QAction::triggered,
                this, &MainWindow::onUnregisterFileAssociations);
#endif
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
        
        // Connect block-level progress signals
        connect(m_worker, &CompressionWorker::totalBlocksChanged,
                this, &MainWindow::onTotalBlocksChanged);
        connect(m_worker, &CompressionWorker::blockProgressChanged,
                this, &MainWindow::onBlockProgressChanged);
        connect(m_worker, &CompressionWorker::blockCompleted,
                this, &MainWindow::onBlockCompleted);
        connect(m_worker, &CompressionWorker::throughputChanged,
                this, &MainWindow::onThroughputChanged);
        connect(m_worker, &CompressionWorker::stageChanged,
                this, &MainWindow::onStageChanged);
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
        
        // Connect block-level progress signals
        connect(m_worker, &CompressionWorker::totalBlocksChanged,
                this, &MainWindow::onTotalBlocksChanged);
        connect(m_worker, &CompressionWorker::blockProgressChanged,
                this, &MainWindow::onBlockProgressChanged);
        connect(m_worker, &CompressionWorker::blockCompleted,
                this, &MainWindow::onBlockCompleted);
        connect(m_worker, &CompressionWorker::throughputChanged,
                this, &MainWindow::onThroughputChanged);
        connect(m_worker, &CompressionWorker::stageChanged,
                this, &MainWindow::onStageChanged);
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
    // Create settings dialog on demand
    if (!m_settingsDialog) {
        m_settingsDialog = new SettingsDialog(this);
        
        // Connect to settings applied signal
        connect(m_settingsDialog, &SettingsDialog::settingsApplied,
                this, &MainWindow::applySettingsToUi);
    } else {
        // Reload settings if dialog already exists
        m_settingsDialog->loadSettings();
    }
    
    // Show the dialog
    m_settingsDialog->exec();
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
    
    // Format: "Progress: XX% (elapsed time)"
    auto formatElapsed = [](qint64 ms) -> QString {
        int seconds = ms / 1000;
        if (seconds < 60) return QString("%1s").arg(seconds);
        int mins = seconds / 60;
        int secs = seconds % 60;
        if (mins < 60) return QString("%1m %2s").arg(mins).arg(secs);
        int hours = mins / 60;
        mins = mins % 60;
        return QString("%1h %2m").arg(hours).arg(mins);
    };
    
    qint64 elapsed = m_worker ? m_worker->getElapsedTime() : 0;
    QString elapsedStr = formatElapsed(elapsed);
    
    if (!currentFile.isEmpty()) {
        ui->progressBar->setFormat(QString("%p% - %1").arg(elapsedStr));
    } else {
        ui->progressBar->setFormat(QString("%p% - %1").arg(elapsedStr));
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
    
    // Include stage in the details if available
    QString stagePrefix = m_currentStage.isEmpty() ? "" : QString("[%1] ").arg(m_currentStage.toUpper());
    QString detailsText = QString("%5%1 / %2 @ %3 MB/s (ETA: %4)")
                             .arg(formatSize(current))
                             .arg(formatSize(total))
                             .arg(speedMBps, 0, 'f', 2)
                             .arg(formatTime(etaSeconds))
                             .arg(stagePrefix);
    
    ui->labelStatus->setText(detailsText);
    
    // Update progress widget if it exists
    if (m_progressWidget) {
        m_progressWidget->setDataProgress(current, total);
        m_progressWidget->updateThroughput(speedMBps);
        m_progressWidget->setETA(etaSeconds);
    }
}

void MainWindow::onWorkerFinished(const QString &outputPath, double compressionRatio, qint64 durationMs)
{
    // Clear current stage
    m_currentStage.clear();
    
    // Keep progress widget open so user can see final result
    // They can close it manually
    
    // Re-enable buttons and restore Compress button text
    ui->buttonCompress->setText("🗜️ Compress");
    ui->buttonCompress->setToolTip("Compress selected files");
    ui->buttonCompress->setEnabled(true);
    // ui->buttonDecompress->setEnabled(true);  // Hidden - archive creation window
    ui->buttonClearFiles->setEnabled(true);
    ui->buttonClearSelected->setEnabled(true);
    
    // Update progress bar to 100%
    ui->progressBar->setValue(100);
    
    if (m_progressWidget) {
        m_progressWidget->updateOverallProgress(1.0f);
    }
    
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
    // Clear current stage
    m_currentStage.clear();
    
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

void MainWindow::applySettingsToUi()
{
    // Create a temporary settings object to read values
    QSettings settings("nvCOMP", "nvCOMP GUI");
    
    // Apply compression settings - default algorithm
    QString defaultAlgorithm = settings.value("compression/defaultAlgorithm", "LZ4").toString();
    int algorithmIndex = ui->comboBoxAlgorithm->findData(defaultAlgorithm);
    if (algorithmIndex >= 0) {
        ui->comboBoxAlgorithm->setCurrentIndex(algorithmIndex);
    }
    
    // Apply compression settings - default volume size (2.5 GB = 2560 MB)
    int defaultVolumeSize = settings.value("compression/defaultVolumeSize", 2560).toInt();
    ui->spinBoxVolumeSize->setValue(defaultVolumeSize);
    
    // Apply compression settings - enable volumes by default
    bool defaultEnableVolumes = settings.value("compression/defaultEnableVolumes", true).toBool();
    ui->checkBoxVolumes->setChecked(defaultEnableVolumes);
    
    // Apply performance settings - CPU/GPU preference
    bool preferGpu = settings.value("performance/preferGpu", true).toBool();
    // If GPU not available, force CPU mode regardless of preference
    if (!m_gpuAvailable) {
        ui->checkBoxCpuMode->setChecked(true);
    } else {
        // Apply the "Prefer CPU" setting to "Force CPU Mode" checkbox
        ui->checkBoxCpuMode->setChecked(!preferGpu);
    }
    
    // Apply interface settings - theme
    QString theme = settings.value("interface/theme", "System").toString();
    applyTheme(theme);
    
    // Note: Other settings like thread count, chunk size, VRAM limit are used by
    // the compression worker internally and don't need UI updates here
    
    statusBar()->showMessage("Settings applied", 2000);
}

void MainWindow::applyTheme(const QString &theme)
{
    if (theme == "Dark") {
        // Apply dark palette
        QPalette darkPalette;
        darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::WindowText, Qt::white);
        darkPalette.setColor(QPalette::Base, QColor(35, 35, 35));
        darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::ToolTipBase, QColor(25, 25, 25));
        darkPalette.setColor(QPalette::ToolTipText, Qt::white);
        darkPalette.setColor(QPalette::Text, Qt::white);
        darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::ButtonText, Qt::white);
        darkPalette.setColor(QPalette::BrightText, Qt::red);
        darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
        darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
        darkPalette.setColor(QPalette::HighlightedText, Qt::black);
        
        // Disabled colors
        darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, QColor(127, 127, 127));
        darkPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(127, 127, 127));
        darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(127, 127, 127));
        
        qApp->setPalette(darkPalette);
        
    } else if (theme == "Light") {
        // Apply light palette (default Qt palette)
        QPalette lightPalette;
        lightPalette.setColor(QPalette::Window, QColor(240, 240, 240));
        lightPalette.setColor(QPalette::WindowText, Qt::black);
        lightPalette.setColor(QPalette::Base, Qt::white);
        lightPalette.setColor(QPalette::AlternateBase, QColor(245, 245, 245));
        lightPalette.setColor(QPalette::ToolTipBase, QColor(255, 255, 220));
        lightPalette.setColor(QPalette::ToolTipText, Qt::black);
        lightPalette.setColor(QPalette::Text, Qt::black);
        lightPalette.setColor(QPalette::Button, QColor(240, 240, 240));
        lightPalette.setColor(QPalette::ButtonText, Qt::black);
        lightPalette.setColor(QPalette::BrightText, Qt::red);
        lightPalette.setColor(QPalette::Link, QColor(0, 0, 255));
        lightPalette.setColor(QPalette::Highlight, QColor(0, 120, 215));
        lightPalette.setColor(QPalette::HighlightedText, Qt::white);
        
        // Disabled colors
        lightPalette.setColor(QPalette::Disabled, QPalette::WindowText, QColor(120, 120, 120));
        lightPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(120, 120, 120));
        lightPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(120, 120, 120));
        
        qApp->setPalette(lightPalette);
        
    } else {
        // "System" - use default system palette
        qApp->setPalette(qApp->style()->standardPalette());
    }
}

void MainWindow::onGPUMonitorTriggered()
{
    // Create GPU monitor on demand
    if (!m_gpuMonitor) {
        m_gpuMonitor = new GPUMonitorWidget(nullptr);  // Top-level window
        m_gpuMonitor->setWindowTitle("GPU Monitor - nvCOMP");
        m_gpuMonitor->setAttribute(Qt::WA_DeleteOnClose, false);
        
        // Connect VRAM warning signal
        connect(m_gpuMonitor, &GPUMonitorWidget::vramLowWarning,
                this, &MainWindow::onVRAMLowWarning);
    }
    
    // Show the GPU monitor window
    m_gpuMonitor->show();
    m_gpuMonitor->raise();
    m_gpuMonitor->activateWindow();
}

void MainWindow::onTotalBlocksChanged(int total)
{
    // DISABLED - Block Progress Widget (Work in Progress)
    // TODO: Re-enable once block progress visualization is stable
    /*
    // Create progress widget dialog on demand
    if (!m_progressWidget) {
        m_progressWidget = new ProgressWidget(nullptr);  // Top-level window
        m_progressWidget->setWindowTitle("*** WORK IN PROGRESS *** Block Progress - nvCOMP");
        m_progressWidget->setWindowFlags(Qt::Dialog);
        m_progressWidget->resize(800, 600);
    }
    
    m_progressWidget->setTotalBlocks(total);
    m_progressWidget->show();
    m_progressWidget->raise();
    */
    
    // Debug output
    qDebug() << "Total blocks set:" << total;
}

void MainWindow::onBlockProgressChanged(int block, float progress)
{
    // DISABLED - Block Progress Widget (Work in Progress)
    /*
    if (m_progressWidget) {
        m_progressWidget->updateBlockProgress(block, progress);
    }
    */
    qDebug() << "Block" << block << "progress:" << (progress * 100) << "%";
}

void MainWindow::onBlockCompleted(int block, float ratio)
{
    // DISABLED - Block Progress Widget (Work in Progress)
    /*
    if (m_progressWidget) {
        m_progressWidget->setBlockComplete(block, ratio);
    }
    */
    qDebug() << "Block" << block << "completed with ratio:" << ratio;
}

void MainWindow::onThroughputChanged(double mbps)
{
    // DISABLED - Block Progress Widget (Work in Progress)
    /*
    if (m_progressWidget) {
        m_progressWidget->updateThroughput(mbps);
    }
    */
    
    // Update status bar with throughput
    statusBar()->showMessage(QString("Speed: %1 MB/s").arg(mbps, 0, 'f', 2));
}

void MainWindow::onStageChanged(const QString &stage)
{
    // Store the current stage
    m_currentStage = stage;
    
    // DISABLED - Block Progress Widget (Work in Progress)
    /*
    if (m_progressWidget) {
        m_progressWidget->setCurrentStage(stage);
    }
    */
}

void MainWindow::onVRAMLowWarning(int deviceIndex, float percentFree)
{
    // Show warning in status bar
    statusBar()->showMessage(
        QString("⚠️ Warning: GPU %1 has low VRAM (%2% free)")
            .arg(deviceIndex)
            .arg(percentFree, 0, 'f', 1),
        10000  // Show for 10 seconds
    );
    
    // Optionally show a message box for critical warnings
    if (percentFree < 5.0f) {
        QMessageBox::warning(this, tr("Low VRAM Warning"),
            tr("GPU %1 is critically low on VRAM (%2% free).\n\n"
               "Compression performance may be degraded or fail.\n"
               "Consider using CPU mode or reducing file size.")
                .arg(deviceIndex)
                .arg(percentFree, 0, 'f', 1));
    }
}

// ============================================================================
// Command-line argument handling
// ============================================================================

void MainWindow::addFilesFromCommandLine(const QStringList &files)
{
    // Add files passed via command-line arguments
    // This is used when the application is launched from context menu
    if (!files.isEmpty()) {
        addFiles(files);
        statusBar()->showMessage(
            tr("Added %1 file(s) from command line").arg(files.count()), 
            3000
        );
    }
}

void MainWindow::setAlgorithmFromCommandLine(const QString &algorithm)
{
    // Map algorithm string to combo box index
    // Algorithm names are stored as userData in the combo box items
    QString algoUpper = algorithm.toUpper();
    
    // Try to find the algorithm by userData
    for (int i = 0; i < ui->comboBoxAlgorithm->count(); ++i) {
        QString itemAlgo = ui->comboBoxAlgorithm->itemData(i).toString();
        if (itemAlgo.compare(algorithm, Qt::CaseInsensitive) == 0 ||
            itemAlgo.compare(algoUpper, Qt::CaseInsensitive) == 0) {
            ui->comboBoxAlgorithm->setCurrentIndex(i);
            statusBar()->showMessage(
                tr("Algorithm set to: %1").arg(itemAlgo), 
                3000
            );
            return;
        }
    }
    
    // Fallback: try to match by display text
    for (int i = 0; i < ui->comboBoxAlgorithm->count(); ++i) {
        QString itemText = ui->comboBoxAlgorithm->itemText(i);
        if (itemText.startsWith(algorithm, Qt::CaseInsensitive) ||
            itemText.startsWith(algoUpper, Qt::CaseInsensitive)) {
            ui->comboBoxAlgorithm->setCurrentIndex(i);
            statusBar()->showMessage(
                tr("Algorithm set to: %1").arg(itemText), 
                3000
            );
            return;
        }
    }
    
    // If not found, show warning
    QMessageBox::warning(this, tr("Unknown Algorithm"),
        tr("Could not find algorithm: %1\n\nUsing default algorithm.").arg(algorithm));
}

void MainWindow::startCompressionFromCommandLine()
{
    // Auto-start compression (called after window is shown and event loop is running)
    // This is used when --compress flag is passed on command line
    
    if (m_fileList.isEmpty()) {
        QMessageBox::warning(this, tr("No Files"),
            tr("Cannot start compression: no files selected."));
        return;
    }
    
    // Start compression with current settings
    onCompressClicked();
}

// ============================================================================
// Windows Context Menu Registration (Windows only)
// ============================================================================

#ifdef _WIN32
void MainWindow::onRegisterContextMenu()
{
    // Check if already registered
    if (ContextMenuManager::isRegistered()) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            tr("Context Menu Already Registered"),
            tr("Windows Explorer context menu is already registered.\n\n"
               "Do you want to re-register it (will overwrite existing entries)?"),
            QMessageBox::Yes | QMessageBox::No
        );
        
        if (reply != QMessageBox::Yes) {
            return;
        }
    }
    
    // Check for admin privileges
    if (!ContextMenuManager::isRunningAsAdmin()) {
        QMessageBox::warning(
            this,
            tr("Administrator Privileges Required"),
            tr("Registering Windows Explorer context menu requires administrator privileges.\n\n"
               "Please restart this application as administrator:\n"
               "1. Right-click nvcomp-gui.exe\n"
               "2. Select 'Run as administrator'\n"
               "3. Try again")
        );
        return;
    }
    
    // Get application path
    QString exePath = QCoreApplication::applicationFilePath();
    QString iconPath = exePath;  // Use .exe as icon (Windows will extract the embedded icon)
    
    // Register
    bool success = ContextMenuManager::registerContextMenu(exePath, iconPath);
    
    if (success) {
        QMessageBox::information(
            this,
            tr("Registration Successful"),
            tr("Windows Explorer context menu has been registered successfully!\n\n"
               "You can now right-click files and folders in Windows Explorer\n"
               "and select 'Compress with nvCOMP' to compress them.\n\n"
               "Changes will take effect immediately.")
        );
        statusBar()->showMessage(tr("Context menu registered successfully"), 5000);
    } else {
        QMessageBox::critical(
            this,
            tr("Registration Failed"),
            tr("Failed to register Windows Explorer context menu.\n\n"
               "Error: %1").arg(ContextMenuManager::getLastError())
        );
        statusBar()->showMessage(tr("Context menu registration failed"), 5000);
    }
}

void MainWindow::onUnregisterContextMenu()
{
    // Check if registered
    if (!ContextMenuManager::isRegistered()) {
        QMessageBox::information(
            this,
            tr("Not Registered"),
            tr("Windows Explorer context menu is not currently registered.")
        );
        return;
    }
    
    // Confirm unregistration
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        tr("Confirm Unregistration"),
        tr("Are you sure you want to remove nvCOMP from Windows Explorer context menu?"),
        QMessageBox::Yes | QMessageBox::No
    );
    
    if (reply != QMessageBox::Yes) {
        return;
    }
    
    // Check for admin privileges
    if (!ContextMenuManager::isRunningAsAdmin()) {
        QMessageBox::warning(
            this,
            tr("Administrator Privileges Required"),
            tr("Unregistering Windows Explorer context menu requires administrator privileges.\n\n"
               "Please restart this application as administrator:\n"
               "1. Right-click nvcomp-gui.exe\n"
               "2. Select 'Run as administrator'\n"
               "3. Try again")
        );
        return;
    }
    
    // Unregister
    bool success = ContextMenuManager::unregisterContextMenu();
    
    if (success) {
        QMessageBox::information(
            this,
            tr("Unregistration Successful"),
            tr("Windows Explorer context menu has been removed successfully!\n\n"
               "Context menu entries will no longer appear when you right-click\n"
               "files and folders in Windows Explorer.")
        );
        statusBar()->showMessage(tr("Context menu unregistered successfully"), 5000);
    } else {
        QMessageBox::critical(
            this,
            tr("Unregistration Failed"),
            tr("Failed to unregister Windows Explorer context menu.\n\n"
               "Error: %1").arg(ContextMenuManager::getLastError())
        );
        statusBar()->showMessage(tr("Context menu unregistration failed"), 5000);
    }
}

// ============================================================================
// Windows File Association Registration (Windows only)
// ============================================================================

void MainWindow::onRegisterFileAssociations()
{
    // Check if already registered
    if (FileAssociationManager::areAllAssociated()) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            tr("File Associations Already Registered"),
            tr("All nvCOMP file associations are already registered.\n\n"
               "Do you want to re-register them (will overwrite existing entries)?"),
            QMessageBox::Yes | QMessageBox::No
        );
        
        if (reply != QMessageBox::Yes) {
            return;
        }
    }
    
    // Check for admin privileges
    if (!FileAssociationManager::isRunningAsAdmin()) {
        QMessageBox::warning(
            this,
            tr("Administrator Privileges Required"),
            tr("Registering file associations requires administrator privileges.\n\n"
               "Please restart this application as administrator:\n"
               "1. Right-click nvcomp-gui.exe\n"
               "2. Select 'Run as administrator'\n"
               "3. Try again")
        );
        return;
    }
    
    // Show info about what will be registered
    QList<FileTypeInfo> fileTypes = FileAssociationManager::getSupportedFileTypes();
    QStringList extensions;
    for (const FileTypeInfo &ft : fileTypes) {
        extensions.append(ft.extension);
    }
    
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        tr("Register File Associations"),
        tr("This will associate the following file types with nvCOMP:\n\n%1\n\n"
           "Double-clicking these files will open them in nvCOMP.\n"
           "Custom icons will be displayed in Windows Explorer.\n"
           "Context menu actions (Extract here, Extract to folder) will be added.\n\n"
           "Continue?").arg(extensions.join(", ")),
        QMessageBox::Yes | QMessageBox::No
    );
    
    if (reply != QMessageBox::Yes) {
        return;
    }
    
    // Get application path
    QString exePath = QCoreApplication::applicationFilePath();
    
    // Register
    bool success = FileAssociationManager::registerAllAssociations(exePath);
    
    if (success) {
        QMessageBox::information(
            this,
            tr("Registration Successful"),
            tr("File associations have been registered successfully!\n\n"
               "The following changes have been made:\n"
               "• Double-click compressed files to open in nvCOMP\n"
               "• Custom icons for each compression algorithm\n"
               "• Right-click menu: 'Extract here' and 'Extract to folder'\n\n"
               "Extensions registered:\n%1\n\n"
               "Changes will take effect immediately.").arg(extensions.join(", "))
        );
        statusBar()->showMessage(tr("File associations registered successfully"), 5000);
    } else {
        QMessageBox::critical(
            this,
            tr("Registration Failed"),
            tr("Failed to register file associations.\n\n"
               "Error: %1").arg(FileAssociationManager::getLastError())
        );
        statusBar()->showMessage(tr("File association registration failed"), 5000);
    }
}

void MainWindow::onUnregisterFileAssociations()
{
    // Check if any are registered
    if (!FileAssociationManager::areAllAssociated()) {
        // Check if at least some are registered
        QList<FileTypeInfo> fileTypes = FileAssociationManager::getSupportedFileTypes();
        bool anyRegistered = false;
        for (const FileTypeInfo &ft : fileTypes) {
            if (FileAssociationManager::isAssociated(ft.extension)) {
                anyRegistered = true;
                break;
            }
        }
        
        if (!anyRegistered) {
            QMessageBox::information(
                this,
                tr("Not Registered"),
                tr("No nvCOMP file associations are currently registered.")
            );
            return;
        }
    }
    
    // Check for admin privileges
    if (!FileAssociationManager::isRunningAsAdmin()) {
        QMessageBox::warning(
            this,
            tr("Administrator Privileges Required"),
            tr("Removing file associations requires administrator privileges.\n\n"
               "Please restart this application as administrator:\n"
               "1. Right-click nvcomp-gui.exe\n"
               "2. Select 'Run as administrator'\n"
               "3. Try again")
        );
        return;
    }
    
    // Confirm unregistration
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        tr("Confirm Unregistration"),
        tr("Are you sure you want to remove nvCOMP file associations?\n\n"
           "This will:\n"
           "• Remove nvCOMP as the default program for these files\n"
           "• Remove custom icons\n"
           "• Remove context menu actions\n\n"
           "You can re-register them later from the Tools menu."),
        QMessageBox::Yes | QMessageBox::No
    );
    
    if (reply != QMessageBox::Yes) {
        return;
    }
    
    // Unregister
    bool success = FileAssociationManager::unregisterAllAssociations();
    
    if (success) {
        QMessageBox::information(
            this,
            tr("Unregistration Successful"),
            tr("File associations have been removed successfully!\n\n"
               "nvCOMP will no longer be associated with compressed file types.")
        );
        statusBar()->showMessage(tr("File associations unregistered successfully"), 5000);
    } else {
        QMessageBox::critical(
            this,
            tr("Unregistration Failed"),
            tr("Failed to unregister file associations.\n\n"
               "Error: %1").arg(FileAssociationManager::getLastError())
        );
        statusBar()->showMessage(tr("File association unregistration failed"), 5000);
    }
}

#else
// Stub implementations for non-Windows platforms
void MainWindow::onRegisterContextMenu()
{
    QMessageBox::information(
        this,
        tr("Not Supported"),
        tr("Context menu registration is only supported on Windows.")
    );
}

void MainWindow::onUnregisterContextMenu()
{
    QMessageBox::information(
        this,
        tr("Not Supported"),
        tr("Context menu unregistration is only supported on Windows.")
    );
}

void MainWindow::onRegisterFileAssociations()
{
    QMessageBox::information(
        this,
        tr("Not Supported"),
        tr("File association registration is only supported on Windows.")
    );
}

void MainWindow::onUnregisterFileAssociations()
{
    QMessageBox::information(
        this,
        tr("Not Supported"),
        tr("File association unregistration is only supported on Windows.")
    );
}
#endif


