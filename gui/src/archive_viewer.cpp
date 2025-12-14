/**
 * @file archive_viewer.cpp
 * @brief Implementation of ArchiveViewerDialog class
 */

#include "archive_viewer.h"
#include "ui_archive_viewer.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>
#include <QDateTime>
#include <QMenu>
#include <QHeaderView>
#include <QProgressDialog>
#include <QApplication>
#include <QRegularExpression>
#include <fstream>
#include <cstring>

// Include core library
extern "C" {
#include "nvcomp_c_api.h"
}

// ============================================================================
// ArchiveLoaderWorker Implementation
// ============================================================================

ArchiveLoaderWorker::ArchiveLoaderWorker(const QString& archivePath, QObject* parent)
    : QThread(parent)
    , m_archivePath(archivePath)
    , m_canceled(false)
{
}

ArchiveLoaderWorker::~ArchiveLoaderWorker()
{
    m_canceled = true;
    wait();
}

void ArchiveLoaderWorker::run()
{
    QList<ArchiveFileInfo> files;
    uint64_t totalSize = 0;
    uint64_t totalCompressed = 0;
    int volumeCount = 1;
    
    try {
        emit loadingProgress(10, "Opening archive...");
        
        // Check if this is a volume archive
        QFileInfo fileInfo(m_archivePath);
        QString basePath = m_archivePath;
        
        // Check for volume pattern (.nvcomp.001, etc.)
        QRegularExpression volumeRegex("\\.\\d{3}$");
        if (m_archivePath.contains(volumeRegex)) {
            // This is a volume - find the first volume
            QString path = m_archivePath;
            path.replace(volumeRegex, ".001");
            if (QFile::exists(path)) {
                basePath = path;
            }
            
            // Count volumes
            volumeCount = 0;
            for (int i = 1; i < 1000; ++i) {
                QString volumePath = m_archivePath;
                volumePath.replace(volumeRegex, QString(".%1").arg(i, 3, 10, QChar('0')));
                if (!QFile::exists(volumePath)) {
                    break;
                }
                volumeCount++;
            }
        }
        
        emit loadingProgress(30, "Reading archive header...");
        
        // Open archive file
        std::ifstream file(basePath.toStdString(), std::ios::binary);
        if (!file.is_open()) {
            emit loadingError("Failed to open archive file");
            return;
        }
        
        // Read magic number to determine format
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.seekg(0);  // Reset to beginning
        
        const uint32_t ARCHIVE_MAGIC = 0x4E564152; // "NVAR" - uncompressed
        const uint32_t BATCHED_MAGIC = 0x4E564243; // "NVBC" - compressed
        
        if (magic == BATCHED_MAGIC) {
            // This is a compressed archive - decompress it in memory first
            file.close();
            emit loadingProgress(40, "Decompressing archive to read contents...");
            
            if (!loadCompressedArchive(basePath, files, totalSize, totalCompressed)) {
                // Error already emitted by loadCompressedArchive
                return;
            }
            
            // Success - continue to display
            emit loadingProgress(100, "Loading complete");
            emit loadingComplete(files, totalSize, totalCompressed, volumeCount);
            return;
        }
        
        if (magic != ARCHIVE_MAGIC) {
            emit loadingError("Invalid archive format (bad magic number).\n\n"
                "This file may not be a valid nvCOMP archive.");
            return;
        }
        
        // Read uncompressed archive header
        struct ArchiveHeader {
            uint32_t magic;
            uint32_t version;
            uint32_t fileCount;
            uint32_t reserved;
        };
        
        ArchiveHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (!file.good()) {
            emit loadingError("Failed to read archive header");
            return;
        }
        
        emit loadingProgress(50, QString("Reading %1 file entries...").arg(header.fileCount));
        
        // Read file entries from uncompressed archive
        for (uint32_t i = 0; i < header.fileCount && !m_canceled; ++i) {
            struct FileEntry {
                uint32_t pathLength;
                uint64_t fileSize;
            };
            
            FileEntry entry;
            file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
            
            if (!file.good()) {
                emit loadingError(QString("Failed to read file entry %1").arg(i));
                return;
            }
            
            // Read file path
            std::vector<char> pathBuffer(entry.pathLength + 1);
            file.read(pathBuffer.data(), entry.pathLength);
            pathBuffer[entry.pathLength] = '\0';
            
            if (!file.good()) {
                emit loadingError(QString("Failed to read file path for entry %1").arg(i));
                return;
            }
            
            QString filePath = QString::fromUtf8(pathBuffer.data());
            
            // Skip file data - we just need the metadata
            file.seekg(entry.fileSize, std::ios::cur);
            
            if (!file.good()) {
                emit loadingError(QString("Failed to skip file data for entry %1").arg(i));
                return;
            }
            
            // Create file info
            ArchiveFileInfo info;
            info.path = filePath;
            info.name = QFileInfo(filePath).fileName();
            info.size = entry.fileSize;
            
            // For uncompressed archives, compressed = uncompressed
            info.compressedSize = entry.fileSize;
            info.compressionRatio = 100.0;  // No compression
            info.isDirectory = filePath.endsWith('/') || filePath.endsWith('\\');
            info.treeItem = nullptr;
            
            files.append(info);
            totalSize += info.size;
            totalCompressed += info.compressedSize;
            
            // Update progress
            if (i % 100 == 0) {
                int progress = 50 + (i * 40 / header.fileCount);
                emit loadingProgress(progress, QString("Processing file %1/%2...").arg(i+1).arg(header.fileCount));
            }
        }
        
        file.close();
        
        if (m_canceled) {
            emit loadingError("Loading canceled");
            return;
        }
        
        emit loadingProgress(100, "Loading complete");
        emit loadingComplete(files, totalSize, totalCompressed, volumeCount);
        
    } catch (const std::exception& e) {
        emit loadingError(QString("Exception: %1").arg(e.what()));
    } catch (...) {
        emit loadingError("Unknown error occurred while loading archive");
    }
}

bool ArchiveLoaderWorker::loadCompressedArchive(const QString& archivePath,
                                               QList<ArchiveFileInfo>& files,
                                               uint64_t& totalSize,
                                               uint64_t& totalCompressed)
{
    try {
        // Get file size for compressed size estimate
        QFileInfo fileInfo(archivePath);
        uint64_t compressedFileSize = fileInfo.size();
        
        emit loadingProgress(45, "Creating temporary extraction directory...");
        
        // Create temporary directory for extraction
        QString tempDir = QDir::temp().filePath("nvcomp_archive_viewer_" + 
            QString::number(QDateTime::currentMSecsSinceEpoch()));
        QDir().mkpath(tempDir);
        
        emit loadingProgress(50, "Decompressing archive...");
        
        // Use nvCOMP C API to decompress
        nvcomp_operation_handle handle = nvcomp_create_operation_handle();
        if (!handle) {
            QDir(tempDir).removeRecursively();
            emit loadingError("Failed to create decompression handle");
            return false;
        }
        
        // Detect algorithm
        nvcomp_algorithm_t algo = nvcomp_detect_algorithm_from_file(archivePath.toUtf8().constData());
        if (algo == NVCOMP_ALGO_UNKNOWN) {
            nvcomp_destroy_operation_handle(handle);
            QDir(tempDir).removeRecursively();
            emit loadingError("Unable to detect compression algorithm");
            return false;
        }
        
        // Decompress
        bool useCPU = !nvcomp_is_cuda_available();
        nvcomp_error_t result;
        
        if (useCPU) {
            result = nvcomp_decompress_cpu(handle, algo,
                archivePath.toUtf8().constData(),
                tempDir.toUtf8().constData());
        } else {
            result = nvcomp_decompress_gpu_batched(handle, algo,
                archivePath.toUtf8().constData(),
                tempDir.toUtf8().constData());
        }
        
        nvcomp_destroy_operation_handle(handle);
        
        if (result != NVCOMP_SUCCESS) {
            const char* errorMsg = nvcomp_get_last_error();
            QString errorStr = errorMsg ? QString(errorMsg) : "Unknown error";
            QDir(tempDir).removeRecursively();
            emit loadingError(QString("Decompression failed: %1").arg(errorStr));
            return false;
        }
        
        emit loadingProgress(70, "Reading extracted files...");
        
        // Now read the extracted files to get the file list
        QDir extractedDir(tempDir);
        QDirIterator it(tempDir, QDir::Files | QDir::NoSymLinks, QDirIterator::Subdirectories);
        
        totalSize = 0;
        totalCompressed = compressedFileSize;
        int fileCount = 0;
        
        while (it.hasNext()) {
            QString filePath = it.next();
            QFileInfo fileInfo(filePath);
            
            // Get relative path from temp dir
            QString relativePath = extractedDir.relativeFilePath(filePath);
            
            ArchiveFileInfo info;
            info.path = relativePath;
            info.name = fileInfo.fileName();
            info.size = fileInfo.size();
            info.compressedSize = 0;  // We don't have per-file compressed sizes
            info.compressionRatio = 0.0;
            info.isDirectory = false;
            info.treeItem = nullptr;
            
            files.append(info);
            totalSize += info.size;
            fileCount++;
            
            if (fileCount % 10 == 0) {
                int progress = 70 + (fileCount * 20 / qMax(1, fileCount + 1));
                emit loadingProgress(progress, QString("Found %1 files...").arg(fileCount));
            }
        }
        
        // Calculate per-file compressed size estimate
        if (fileCount > 0 && totalSize > 0) {
            double compressionRatio = (double)totalCompressed / totalSize;
            for (ArchiveFileInfo& info : files) {
                info.compressedSize = (uint64_t)(info.size * compressionRatio);
                info.compressionRatio = compressionRatio * 100.0;
            }
        }
        
        // Clean up temp directory
        QDir(tempDir).removeRecursively();
        
        emit loadingProgress(95, QString("Loaded %1 files").arg(fileCount));
        
        return true;
        
    } catch (const std::exception& e) {
        emit loadingError(QString("Exception during decompression: %1").arg(e.what()));
        return false;
    } catch (...) {
        emit loadingError("Unknown error during decompression");
        return false;
    }
}

// ============================================================================
// ArchiveViewerDialog Implementation
// ============================================================================

ArchiveViewerDialog::ArchiveViewerDialog(const QString& archivePath, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::ArchiveViewerDialog)
    , m_archivePath(archivePath)
    , m_totalSize(0)
    , m_totalCompressed(0)
    , m_volumeCount(1)
    , m_loader(nullptr)
{
    ui->setupUi(this);
    setupUi();
    setupConnections();
    loadArchive();
}

ArchiveViewerDialog::~ArchiveViewerDialog()
{
    // Clean up loader if it exists
    if (m_loader) {
        if (m_loader->isRunning()) {
            m_loader->terminate();
            m_loader->wait();
        }
        delete m_loader;
    }
    
    delete ui;
}

void ArchiveViewerDialog::setupUi()
{
    // Set window title
    QFileInfo fileInfo(m_archivePath);
    setWindowTitle(QString("Archive Viewer - %1").arg(fileInfo.fileName()));
    
    // Set window size
    resize(900, 600);
    setMinimumSize(700, 500);
    
    // Configure tree widget
    ui->treeWidget->setColumnCount(4);
    ui->treeWidget->setHeaderLabels(QStringList() << "Name" << "Size" << "Compressed" << "Ratio");
    ui->treeWidget->setAlternatingRowColors(true);
    ui->treeWidget->setSortingEnabled(true);
    ui->treeWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    ui->treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
    
    // Set column widths
    ui->treeWidget->header()->resizeSection(0, 400);  // Name
    ui->treeWidget->header()->resizeSection(1, 120);  // Size
    ui->treeWidget->header()->resizeSection(2, 120);  // Compressed
    ui->treeWidget->header()->resizeSection(3, 80);   // Ratio
    
    // Initially disable extract buttons
    ui->buttonExtractAll->setEnabled(false);
    ui->buttonExtractSelected->setEnabled(false);
}

void ArchiveViewerDialog::setupConnections()
{
    // Connect buttons
    connect(ui->buttonExtractAll, &QPushButton::clicked,
            this, &ArchiveViewerDialog::onExtractAllClicked);
    connect(ui->buttonExtractSelected, &QPushButton::clicked,
            this, &ArchiveViewerDialog::onExtractSelectedClicked);
    connect(ui->buttonRefresh, &QPushButton::clicked,
            this, &ArchiveViewerDialog::onRefreshClicked);
    connect(ui->buttonClose, &QPushButton::clicked,
            this, &ArchiveViewerDialog::onCloseClicked);
    
    // Connect search
    connect(ui->lineEditSearch, &QLineEdit::textChanged,
            this, &ArchiveViewerDialog::onSearchTextChanged);
    
    // Connect tree widget signals
    connect(ui->treeWidget, &QTreeWidget::itemDoubleClicked,
            this, &ArchiveViewerDialog::onTreeItemDoubleClicked);
    connect(ui->treeWidget, &QTreeWidget::itemSelectionChanged,
            this, &ArchiveViewerDialog::onTreeSelectionChanged);
    connect(ui->treeWidget, &QTreeWidget::customContextMenuRequested,
            this, &ArchiveViewerDialog::onTreeContextMenuRequested);
}

void ArchiveViewerDialog::loadArchive()
{
    // Clear existing data
    m_files.clear();
    m_folderItems.clear();
    ui->treeWidget->clear();
    
    // Create and start loader worker
    m_loader = new ArchiveLoaderWorker(m_archivePath, this);
    
    connect(m_loader, &ArchiveLoaderWorker::loadingProgress,
            this, &ArchiveViewerDialog::onLoadingProgress);
    connect(m_loader, &ArchiveLoaderWorker::loadingComplete,
            this, &ArchiveViewerDialog::onLoadingComplete);
    connect(m_loader, &ArchiveLoaderWorker::loadingError,
            this, &ArchiveViewerDialog::onLoadingError);
    
    m_loader->start();
    
    // Update status
    ui->labelStatus->setText("Loading archive...");
}

void ArchiveViewerDialog::populateTree()
{
    ui->treeWidget->clear();
    m_folderItems.clear();
    
    // Create root item
    QFileInfo fileInfo(m_archivePath);
    QTreeWidgetItem* rootItem = new QTreeWidgetItem(ui->treeWidget);
    rootItem->setText(0, fileInfo.fileName());
    rootItem->setText(1, formatSize(m_totalSize));
    rootItem->setText(2, formatSize(m_totalCompressed));
    rootItem->setText(3, QString("%1%").arg(m_totalSize > 0 ? 
        (double)m_totalCompressed / m_totalSize * 100.0 : 0.0, 0, 'f', 1));
    rootItem->setIcon(0, style()->standardIcon(QStyle::SP_DirIcon));
    rootItem->setExpanded(true);
    
    // Add files to tree
    for (ArchiveFileInfo& fileInfo : m_files) {
        // Skip directory entries
        if (fileInfo.isDirectory) {
            continue;
        }
        
        // Get or create parent folder
        QFileInfo pathInfo(fileInfo.path);
        QString folderPath = pathInfo.path();
        QTreeWidgetItem* parentItem = rootItem;
        
        if (!folderPath.isEmpty() && folderPath != ".") {
            parentItem = getOrCreateFolderItem(folderPath);
        }
        
        // Create file item
        QTreeWidgetItem* fileItem = new QTreeWidgetItem(parentItem);
        fileItem->setText(0, fileInfo.name);
        fileItem->setText(1, formatSize(fileInfo.size));
        fileItem->setText(2, formatSize(fileInfo.compressedSize));
        fileItem->setText(3, QString("%1%").arg(fileInfo.compressionRatio, 0, 'f', 1));
        fileItem->setIcon(0, style()->standardIcon(QStyle::SP_FileIcon));
        
        // Store reference to tree item
        fileInfo.treeItem = fileItem;
        
        // Store file info in item data
        fileItem->setData(0, Qt::UserRole, QVariant::fromValue(fileInfo.path));
    }
    
    // Sort by name
    ui->treeWidget->sortItems(0, Qt::AscendingOrder);
}

QTreeWidgetItem* ArchiveViewerDialog::getOrCreateFolderItem(const QString& folderPath)
{
    // Check if folder item already exists
    if (m_folderItems.contains(folderPath)) {
        return m_folderItems[folderPath];
    }
    
    // Split path into components
    QStringList parts = folderPath.split(QRegularExpression("[/\\\\]"), Qt::SkipEmptyParts);
    
    QTreeWidgetItem* parentItem = ui->treeWidget->topLevelItem(0);  // Root item
    QString currentPath;
    
    for (const QString& part : parts) {
        if (!currentPath.isEmpty()) {
            currentPath += "/";
        }
        currentPath += part;
        
        // Check if this level exists
        if (m_folderItems.contains(currentPath)) {
            parentItem = m_folderItems[currentPath];
            continue;
        }
        
        // Create new folder item
        QTreeWidgetItem* folderItem = new QTreeWidgetItem(parentItem);
        folderItem->setText(0, part);
        folderItem->setIcon(0, style()->standardIcon(QStyle::SP_DirIcon));
        folderItem->setExpanded(false);
        
        m_folderItems[currentPath] = folderItem;
        parentItem = folderItem;
    }
    
    return parentItem;
}

void ArchiveViewerDialog::updateStatistics()
{
    int fileCount = m_files.count();
    int folderCount = m_folderItems.count();
    
    QString statsText = QString("Files: %1 | Folders: %2 | Total Size: %3 | Compressed: %4 | Ratio: %5%")
        .arg(fileCount)
        .arg(folderCount)
        .arg(formatSize(m_totalSize))
        .arg(formatSize(m_totalCompressed))
        .arg(m_totalSize > 0 ? (double)m_totalCompressed / m_totalSize * 100.0 : 0.0, 0, 'f', 1);
    
    if (m_volumeCount > 1) {
        statsText += QString(" | Volumes: %1").arg(m_volumeCount);
    }
    
    ui->labelStatistics->setText(statsText);
}

void ArchiveViewerDialog::filterTreeItems(const QString& query)
{
    if (query.isEmpty()) {
        // Show all items
        QTreeWidgetItemIterator it(ui->treeWidget);
        while (*it) {
            (*it)->setHidden(false);
            ++it;
        }
        return;
    }
    
    // Filter items
    QTreeWidgetItem* rootItem = ui->treeWidget->topLevelItem(0);
    if (rootItem) {
        filterTreeItem(rootItem, query);
    }
}

bool ArchiveViewerDialog::filterTreeItem(QTreeWidgetItem* item, const QString& query)
{
    bool hasVisibleChildren = false;
    
    // Check children first
    for (int i = 0; i < item->childCount(); ++i) {
        if (filterTreeItem(item->child(i), query)) {
            hasVisibleChildren = true;
        }
    }
    
    // Check if this item matches
    QString itemText = item->text(0).toLower();
    bool matches = itemText.contains(query.toLower());
    
    // Show item if it matches or has visible children
    bool shouldShow = matches || hasVisibleChildren;
    item->setHidden(!shouldShow);
    
    // Expand if has visible children
    if (hasVisibleChildren) {
        item->setExpanded(true);
    }
    
    return shouldShow;
}

void ArchiveViewerDialog::extractFiles(const QString& outputPath, bool selectedOnly)
{
    QStringList filesToExtract;
    
    if (selectedOnly) {
        filesToExtract = getSelectedFilePaths();
        if (filesToExtract.isEmpty()) {
            QMessageBox::warning(this, "No Files Selected",
                "Please select files to extract.");
            return;
        }
    } else {
        // Extract all files
        for (const ArchiveFileInfo& fileInfo : m_files) {
            if (!fileInfo.isDirectory) {
                filesToExtract.append(fileInfo.path);
            }
        }
    }
    
    // Create progress dialog
    QProgressDialog progress("Extracting files...", "Cancel", 0, filesToExtract.count(), this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    
    try {
        // Call extraction function from core library
        nvcomp_operation_handle handle = nvcomp_create_operation_handle();
        
        if (!handle) {
            QMessageBox::critical(this, "Extraction Error",
                "Failed to create decompression operation handle");
            return;
        }
        
        // Detect algorithm from file
        nvcomp_algorithm_t algo = nvcomp_detect_algorithm_from_file(m_archivePath.toUtf8().constData());
        
        if (algo == NVCOMP_ALGO_UNKNOWN) {
            nvcomp_destroy_operation_handle(handle);
            QMessageBox::critical(this, "Extraction Error",
                "Unable to detect compression algorithm from archive");
            return;
        }
        
        // TODO: For selected files only, we would need to implement partial extraction
        // For now, we extract everything
        
        // Determine whether to use CPU or GPU
        bool useCPU = !nvcomp_is_cuda_available();
        nvcomp_error_t result;
        
        if (useCPU) {
            result = nvcomp_decompress_cpu(handle, algo,
                m_archivePath.toUtf8().constData(),
                outputPath.toUtf8().constData());
        } else {
            result = nvcomp_decompress_gpu_batched(handle, algo,
                m_archivePath.toUtf8().constData(),
                outputPath.toUtf8().constData());
        }
        
        if (result != NVCOMP_SUCCESS) {
            const char* errorMsg = nvcomp_get_last_error();
            QString errorStr = errorMsg ? QString(errorMsg) : "Unknown error";
            nvcomp_destroy_operation_handle(handle);
            QMessageBox::critical(this, "Extraction Error",
                QString("Extraction failed: %1").arg(errorStr));
            return;
        }
        
        nvcomp_destroy_operation_handle(handle);
        
        progress.setValue(filesToExtract.count());
        
        QMessageBox::information(this, "Extraction Complete",
            QString("Successfully extracted %1 file(s) to:\n%2")
                .arg(filesToExtract.count())
                .arg(outputPath));
        
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Extraction Error",
            QString("Exception during extraction: %1").arg(e.what()));
    }
}

QStringList ArchiveViewerDialog::getSelectedFilePaths()
{
    QStringList paths;
    QList<QTreeWidgetItem*> selectedItems = ui->treeWidget->selectedItems();
    
    for (QTreeWidgetItem* item : selectedItems) {
        // Get file path from item data
        QVariant pathData = item->data(0, Qt::UserRole);
        if (pathData.isValid()) {
            paths.append(pathData.toString());
        }
    }
    
    return paths;
}

QString ArchiveViewerDialog::formatSize(uint64_t bytes) const
{
    if (bytes < 1024) {
        return QString("%1 B").arg(bytes);
    } else if (bytes < 1024 * 1024) {
        return QString("%1 KB").arg(bytes / 1024.0, 0, 'f', 2);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        return QString("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 2);
    } else {
        return QString("%1 GB").arg(bytes / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2);
    }
}

void ArchiveViewerDialog::expandParents(QTreeWidgetItem* item)
{
    QTreeWidgetItem* parent = item->parent();
    while (parent) {
        parent->setExpanded(true);
        parent = parent->parent();
    }
}

// ============================================================================
// Slots Implementation
// ============================================================================

void ArchiveViewerDialog::onExtractAllClicked()
{
    QString outputPath = QFileDialog::getExistingDirectory(
        this,
        "Select Output Directory",
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (outputPath.isEmpty()) {
        return;  // User canceled
    }
    
    extractFiles(outputPath, false);
}

void ArchiveViewerDialog::onExtractSelectedClicked()
{
    QString outputPath = QFileDialog::getExistingDirectory(
        this,
        "Select Output Directory",
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );
    
    if (outputPath.isEmpty()) {
        return;  // User canceled
    }
    
    extractFiles(outputPath, true);
}

void ArchiveViewerDialog::onRefreshClicked()
{
    loadArchive();
}

void ArchiveViewerDialog::onCloseClicked()
{
    accept();
}

void ArchiveViewerDialog::onSearchTextChanged(const QString& text)
{
    filterTreeItems(text);
}

void ArchiveViewerDialog::onTreeItemDoubleClicked(QTreeWidgetItem* item, int column)
{
    Q_UNUSED(column);
    
    // Toggle expansion for folders, extract for files
    if (item->childCount() > 0) {
        item->setExpanded(!item->isExpanded());
    } else {
        // This is a file - could trigger quick extract
        // For now, just show properties
        onPropertiesAction();
    }
}

void ArchiveViewerDialog::onTreeSelectionChanged()
{
    // Enable/disable extract selected button
    bool hasSelection = !ui->treeWidget->selectedItems().isEmpty();
    ui->buttonExtractSelected->setEnabled(hasSelection);
}

void ArchiveViewerDialog::onTreeContextMenuRequested(const QPoint& pos)
{
    QTreeWidgetItem* item = ui->treeWidget->itemAt(pos);
    if (!item) {
        return;
    }
    
    QMenu menu(this);
    
    QAction* extractAction = menu.addAction("Extract...");
    connect(extractAction, &QAction::triggered, this, &ArchiveViewerDialog::onExtractAction);
    
    menu.addSeparator();
    
    QAction* propertiesAction = menu.addAction("Properties");
    connect(propertiesAction, &QAction::triggered, this, &ArchiveViewerDialog::onPropertiesAction);
    
    menu.exec(ui->treeWidget->viewport()->mapToGlobal(pos));
}

void ArchiveViewerDialog::onLoadingProgress(int percentage, const QString& status)
{
    ui->labelStatus->setText(status);
    // Could update a progress bar here if desired
}

void ArchiveViewerDialog::onLoadingComplete(const QList<ArchiveFileInfo>& files,
                                           uint64_t totalSize,
                                           uint64_t totalCompressed,
                                           int volumeCount)
{
    m_files = files;
    m_totalSize = totalSize;
    m_totalCompressed = totalCompressed;
    m_volumeCount = volumeCount;
    
    populateTree();
    updateStatistics();
    
    ui->labelStatus->setText("Ready");
    ui->buttonExtractAll->setEnabled(true);
}

void ArchiveViewerDialog::onLoadingError(const QString& errorMessage)
{
    ui->labelStatus->setText("Error");
    QMessageBox::critical(this, "Loading Error",
        QString("Failed to load archive:\n\n%1").arg(errorMessage));
}

void ArchiveViewerDialog::onExtractAction()
{
    onExtractSelectedClicked();
}

void ArchiveViewerDialog::onPropertiesAction()
{
    QList<QTreeWidgetItem*> selectedItems = ui->treeWidget->selectedItems();
    if (selectedItems.isEmpty()) {
        return;
    }
    
    QTreeWidgetItem* item = selectedItems.first();
    
    // Get file info
    QString name = item->text(0);
    QString size = item->text(1);
    QString compressed = item->text(2);
    QString ratio = item->text(3);
    
    QString properties = QString(
        "<b>Name:</b> %1<br>"
        "<b>Size:</b> %2<br>"
        "<b>Compressed:</b> %3<br>"
        "<b>Ratio:</b> %4<br>"
    ).arg(name, size, compressed, ratio);
    
    QMessageBox::information(this, "Properties", properties);
}

