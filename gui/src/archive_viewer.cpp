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
#include <QDateTime>
#include <QMenu>
#include <QHeaderView>
#include <QProgressDialog>
#include <QApplication>
#include <QRegularExpression>
#include <QTimer>
#include <fstream>
#include <cstring>

// Include core library
extern "C" {
#include "nvcomp_c_api.h"
}

// ============================================================================
// Listing callbacks (nvcomp_list_archive_entries)
// ============================================================================

namespace {

struct ListContext {
    ArchiveLoaderWorker* worker;
    QList<ArchiveFileInfo>* files;
    uint64_t totalSize = 0;
};

// Called once per archive entry, on the worker thread. Signals emitted here
// cross to the GUI thread via queued connections.
int viewerEntryCallback(const char* path, uint64_t size, uint32_t mode,
                        uint64_t mtimeNs, void* userData)
{
    auto* ctx = static_cast<ListContext*>(userData);
    if (ctx->worker->isCanceled()) {
        return 1;  // cancel the listing
    }

    ArchiveFileInfo info;
    info.path = QString::fromUtf8(path);
    info.path.replace(QLatin1Char('\\'), QLatin1Char('/'));  // Windows-built archives
    int slash = info.path.lastIndexOf(QLatin1Char('/'));
    info.name = slash >= 0 ? info.path.mid(slash + 1) : info.path;
    info.size = size;
    info.compressedSize = 0;
    info.compressionRatio = 0.0;
    info.isDirectory = info.path.endsWith(QLatin1Char('/'));
    info.treeItem = nullptr;
    info.mode = mode;
    info.mtimeNs = mtimeNs;

    ctx->files->append(info);
    ctx->totalSize += size;
    return 0;
}

// Byte-level progress (one call per decompressed sub-batch): map onto the
// 45-90% band of the dialog's progress scale.
void viewerProgressCallback(uint64_t current, uint64_t total, void* userData)
{
    auto* ctx = static_cast<ListContext*>(userData);
    int pct = 45;
    if (total > 0) {
        pct += static_cast<int>((static_cast<double>(current) / total) * 45.0);
    }
    emit ctx->worker->loadingProgress(qMin(pct, 90),
        QString("Reading archive contents... %1 files found")
            .arg(ctx->files->count()));
}

} // namespace

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
        uint64_t compressedBytesOnDisk = 0;

        // Volume naming convention: stem.vol001.ext (see core volume.cpp).
        QRegularExpression volumeRegex("\\.vol(\\d{3})\\.");
        if (m_archivePath.contains(volumeRegex)) {
            // This is a volume - route to the first one (it holds the manifest).
            QString path = m_archivePath;
            path.replace(volumeRegex, ".vol001.");
            if (QFile::exists(path)) {
                basePath = path;
            }

            // Count sibling volumes (header display) and sum their on-disk
            // sizes (compressed-total statistic covers every volume, not
            // just the one that was opened).
            volumeCount = 0;
            for (int i = 1; i < 1000; ++i) {
                QString volumePath = m_archivePath;
                volumePath.replace(volumeRegex,
                                   QString(".vol%1.").arg(i, 3, 10, QChar('0')));
                QFileInfo volumeInfo(volumePath);
                if (!volumeInfo.exists()) {
                    break;
                }
                compressedBytesOnDisk += volumeInfo.size();
                volumeCount++;
            }
            if (volumeCount == 0) volumeCount = 1;
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
        const uint32_t VOLUME_MAGIC = 0x4E56564D;  // "NVVM" - multi-volume manifest

        if (magic == BATCHED_MAGIC || magic == VOLUME_MAGIC) {
            // Compressed (or multi-volume) archive: stream-list the entries
            // via the metadata API -- nothing is extracted to disk.
            file.close();
            emit loadingProgress(40, "Reading archive contents...");

            if (compressedBytesOnDisk == 0) {
                compressedBytesOnDisk = QFileInfo(basePath).size();
            }
            if (!loadCompressedArchive(basePath, files, totalSize,
                                       compressedBytesOnDisk)) {
                // Error already emitted by loadCompressedArchive
                return;
            }
            totalCompressed = compressedBytesOnDisk;

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

        // v1 entries are 16 bytes (pathLength + padding + fileSize); v2 adds
        // mode + mtime (24 bytes). Must match core/include/nvcomp_core.hpp.
        if (header.version < 1 || header.version > 2) {
            emit loadingError(QString("Unsupported archive version (%1).\n\n"
                "This archive was created by a newer version of nvCOMP.")
                .arg(header.version));
            return;
        }
        const bool v1 = header.version < 2;

        emit loadingProgress(50, QString("Reading %1 file entries...").arg(header.fileCount));

        // Read file entries from uncompressed archive
        for (uint32_t i = 0; i < header.fileCount && !m_canceled; ++i) {
            struct FileEntryV1 {
                uint32_t pathLength;
                uint64_t fileSize;
            };
            struct FileEntry {
                uint32_t pathLength;
                uint32_t mode;
                uint64_t fileSize;
                uint64_t mtimeNs;
            };

            FileEntry entry;
            if (v1) {
                FileEntryV1 e1;
                file.read(reinterpret_cast<char*>(&e1), sizeof(e1));
                entry.pathLength = e1.pathLength;
                entry.mode = 0;
                entry.fileSize = e1.fileSize;
                entry.mtimeNs = 0;
            } else {
                file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
            }

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
                                               uint64_t totalCompressed)
{
    try {
        emit loadingProgress(45, "Reading archive contents...");

        // Stream the entry metadata out of the archive: the decompressed
        // bytes are parsed on the fly and discarded, so no temp directory,
        // no disk writes, sub-batch memory use.
        ListContext ctx;
        ctx.worker = this;
        ctx.files = &files;

        nvcomp_error_t result = nvcomp_list_archive_entries(
            archivePath.toUtf8().constData(),
            viewerEntryCallback,
            viewerProgressCallback,
            &ctx);

        if (result == NVCOMP_ERROR_CANCELED) {
            emit loadingError("Loading canceled");
            return false;
        }
        if (result != NVCOMP_SUCCESS) {
            const char* errorMsg = nvcomp_get_last_error();
            emit loadingError(QString("Failed to read archive contents: %1")
                .arg(errorMsg && *errorMsg ? QString(errorMsg) : "Unknown error"));
            return false;
        }

        totalSize = ctx.totalSize;

        // Per-file compressed sizes aren't stored in the archive; estimate
        // from the global ratio (same presentation as before).
        if (totalSize > 0) {
            double compressionRatio = (double)totalCompressed / totalSize;
            for (ArchiveFileInfo& info : files) {
                info.compressedSize = (uint64_t)(info.size * compressionRatio);
                info.compressionRatio = compressionRatio * 100.0;
            }
        }

        emit loadingProgress(95, QString("Loaded %1 files").arg(files.count()));
        return true;

    } catch (const std::exception& e) {
        emit loadingError(QString("Exception while reading archive: %1").arg(e.what()));
        return false;
    } catch (...) {
        emit loadingError("Unknown error while reading archive");
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
    , m_searchDebounce(nullptr)
{
    ui->setupUi(this);
    setupUi();
    setupConnections();
    loadArchive();
}

ArchiveViewerDialog::~ArchiveViewerDialog()
{
    // Clean up loader if it exists. Cooperative cancel + wait: terminate()
    // would kill a thread that may be inside the CUDA driver.
    if (m_loader) {
        if (m_loader->isRunning()) {
            m_loader->cancel();
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
    // Sorting stays off until populateTree() finishes: with sorting active,
    // every insertion re-sorts, which hangs the UI on 100k-file archives.
    ui->treeWidget->setUniformRowHeights(true);
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
    
    // Connect search through a debounce timer: filtering walks every tree
    // item, which is too slow to run per keystroke on 100k-file archives.
    m_searchDebounce = new QTimer(this);
    m_searchDebounce->setSingleShot(true);
    m_searchDebounce->setInterval(250);
    connect(m_searchDebounce, &QTimer::timeout, this, [this]() {
        ui->treeWidget->setUpdatesEnabled(false);
        filterTreeItems(ui->lineEditSearch->text());
        ui->treeWidget->setUpdatesEnabled(true);
    });
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
    // Stop and drop any previous loader (Refresh re-enters here).
    if (m_loader) {
        if (m_loader->isRunning()) {
            m_loader->cancel();
            m_loader->wait();
        }
        delete m_loader;
        m_loader = nullptr;
    }

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
    // Bulk build: updates and sorting off, whole tree assembled detached,
    // attached with a single addTopLevelItem, then one sort at the end.
    // Anything else is quadratic at 100k items.
    ui->treeWidget->setUpdatesEnabled(false);
    ui->treeWidget->setSortingEnabled(false);
    ui->treeWidget->clear();
    m_folderItems.clear();

    const QIcon dirIcon = style()->standardIcon(QStyle::SP_DirIcon);
    const QIcon fileIcon = style()->standardIcon(QStyle::SP_FileIcon);

    // Create root item (detached until the tree is fully built)
    QFileInfo fileInfo(m_archivePath);
    QTreeWidgetItem* rootItem = new QTreeWidgetItem();
    rootItem->setText(0, fileInfo.fileName());
    rootItem->setText(1, formatSize(m_totalSize));
    rootItem->setText(2, formatSize(m_totalCompressed));
    rootItem->setText(3, QString("%1%").arg(m_totalSize > 0 ?
        (double)m_totalCompressed / m_totalSize * 100.0 : 0.0, 0, 'f', 1));
    rootItem->setIcon(0, dirIcon);

    // Add files to tree
    for (ArchiveFileInfo& info : m_files) {
        // Skip directory entries
        if (info.isDirectory) {
            continue;
        }

        // Get or create parent folder
        int slash = info.path.lastIndexOf(QLatin1Char('/'));
        QTreeWidgetItem* parentItem = rootItem;
        if (slash > 0) {
            parentItem = getOrCreateFolderItem(info.path.left(slash), rootItem, dirIcon);
        }

        // Create file item
        QTreeWidgetItem* fileItem = new QTreeWidgetItem(parentItem);
        fileItem->setText(0, info.name);
        fileItem->setText(1, formatSize(info.size));
        fileItem->setText(2, formatSize(info.compressedSize));
        fileItem->setText(3, QString("%1%").arg(info.compressionRatio, 0, 'f', 1));
        fileItem->setIcon(0, fileIcon);

        // Store reference to tree item
        info.treeItem = fileItem;

        // Store file info in item data
        fileItem->setData(0, Qt::UserRole, QVariant::fromValue(info.path));
    }

    // Single attach of the whole tree, then one sort.
    ui->treeWidget->addTopLevelItem(rootItem);
    ui->treeWidget->sortItems(0, Qt::AscendingOrder);
    ui->treeWidget->setSortingEnabled(true);
    ui->treeWidget->setUpdatesEnabled(true);

    // Expand the root and any single-child folder chain below it, so an
    // archive wrapping one top-level folder opens straight to its contents.
    rootItem->setExpanded(true);
    for (QTreeWidgetItem* item = rootItem;
         item->childCount() == 1 && item->child(0)->childCount() > 0;
         item = item->child(0)) {
        item->child(0)->setExpanded(true);
    }
}

QTreeWidgetItem* ArchiveViewerDialog::getOrCreateFolderItem(const QString& folderPath,
                                                            QTreeWidgetItem* rootItem,
                                                            const QIcon& dirIcon)
{
    // Check if folder item already exists
    auto it = m_folderItems.constFind(folderPath);
    if (it != m_folderItems.constEnd()) {
        return it.value();
    }

    // Split path into components
    QStringList parts = folderPath.split(QLatin1Char('/'), Qt::SkipEmptyParts);

    QTreeWidgetItem* parentItem = rootItem;
    QString currentPath;

    for (const QString& part : parts) {
        if (!currentPath.isEmpty()) {
            currentPath += "/";
        }
        currentPath += part;

        // Check if this level exists
        auto found = m_folderItems.constFind(currentPath);
        if (found != m_folderItems.constEnd()) {
            parentItem = found.value();
            continue;
        }

        // Create new folder item
        QTreeWidgetItem* folderItem = new QTreeWidgetItem(parentItem);
        folderItem->setText(0, part);
        folderItem->setIcon(0, dirIcon);

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
                outputPath.toUtf8().constData(),
                nullptr);
        } else {
            result = nvcomp_decompress_gpu_batched(handle, algo,
                m_archivePath.toUtf8().constData(),
                outputPath.toUtf8().constData(),
                nullptr);
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
    Q_UNUSED(text);
    m_searchDebounce->start();  // restart the 250 ms debounce window
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

