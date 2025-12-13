/**
 * @file mainwindow.cpp
 * @brief Implementation of MainWindow class
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"
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
    connect(ui->buttonDecompress, &QPushButton::clicked,
            this, &MainWindow::onDecompressClicked);
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
    // For now, simulate GPU check
    // In future tasks, this will use actual nvCOMP GPU detection
    m_gpuAvailable = false;  // Assume no GPU for now
    
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
    // Enable/disable compress and decompress buttons based on file list
    bool hasFiles = !m_fileList.isEmpty();
    ui->buttonCompress->setEnabled(hasFiles);
    ui->buttonDecompress->setEnabled(hasFiles);
    
    // Update status label
    if (hasFiles) {
        ui->labelStatus->setText(QString("Ready - %1 file(s) selected").arg(m_fileList.count()));
    } else {
        ui->labelStatus->setText("Ready - No files selected");
    }
}

void MainWindow::addFiles(const QStringList &files)
{
    for (const QString &filePath : files) {
        QFileInfo fileInfo(filePath);
        
        if (fileInfo.isFile()) {
            // Skip if already in list
            if (m_fileList.contains(filePath)) {
                continue;
            }
            
            // Add file to internal list
            m_fileList.append(filePath);
            
            // Add to UI list widget
            ui->listWidgetFiles->addItem(fileInfo.absoluteFilePath());
        }
        else if (fileInfo.isDir()) {
            // Handle directory - add all files from it (non-recursive for now)
            QDir dir(filePath);
            QFileInfoList fileList = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
            
            for (const QFileInfo &file : fileList) {
                QString absPath = file.absoluteFilePath();
                
                // Skip if already in list
                if (m_fileList.contains(absPath)) {
                    continue;
                }
                
                // Add file to internal list
                m_fileList.append(absPath);
                
                // Add to UI list widget with folder context
                QString displayText = QString("%1/%2").arg(fileInfo.fileName()).arg(file.fileName());
                ui->listWidgetFiles->addItem(absPath);
            }
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
        QStringList filesToAdd;
        
        for (const QUrl &url : mimeData->urls()) {
            QString filePath = url.toLocalFile();
            QFileInfo fileInfo(filePath);
            
            if (fileInfo.exists()) {
                if (fileInfo.isFile()) {
                    filesToAdd.append(filePath);
                } else if (fileInfo.isDir()) {
                    // Add all files from directory (non-recursive for now)
                    QDir dir(filePath);
                    QFileInfoList fileList = dir.entryInfoList(QDir::Files);
                    for (const QFileInfo &file : fileList) {
                        filesToAdd.append(file.absoluteFilePath());
                    }
                }
            }
        }
        
        if (!filesToAdd.isEmpty()) {
            addFiles(filesToAdd);
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
    return ui->comboBoxAlgorithm->currentText();
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
    // Stub for now - will be implemented in future tasks
    QString outputName = getOutputArchiveName();
    if (outputName.isEmpty()) {
        outputName = "(auto-generated)";
    }
    
    QMessageBox::information(this,
        tr("Compress"),
        tr("Compression will be implemented in Task 2.3: Compression Backend Integration\n\n"
           "Current settings:\n"
           "Algorithm: %1\n"
           "Files: %2\n"
           "Output: %3\n"
           "CPU Mode: %4")
           .arg(getSelectedAlgorithm())
           .arg(m_fileList.count())
           .arg(outputName)
           .arg(isCpuModeEnabled() ? "Yes" : "No")
    );
}

void MainWindow::onDecompressClicked()
{
    // Stub for now - will be implemented in future tasks
    QMessageBox::information(this,
        tr("Decompress"),
        tr("Decompression will be implemented in Task 2.3: Compression Backend Integration\n\n"
           "Current settings:\n"
           "Files: %1\n"
           "CPU Mode: %2")
           .arg(m_fileList.count())
           .arg(isCpuModeEnabled() ? "Yes" : "No")
    );
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


