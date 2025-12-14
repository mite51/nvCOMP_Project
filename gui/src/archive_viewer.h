/**
 * @file archive_viewer.h
 * @brief Archive viewer dialog for displaying and extracting archive contents
 * 
 * Provides a dialog window that displays the contents of a compressed archive
 * in a tree view, similar to WinRAR or 7-Zip, with extraction capabilities.
 */

#ifndef ARCHIVE_VIEWER_H
#define ARCHIVE_VIEWER_H

#include <QDialog>
#include <QString>
#include <QStringList>
#include <QTreeWidgetItem>
#include <QMap>
#include <QThread>

QT_BEGIN_NAMESPACE
namespace Ui { class ArchiveViewerDialog; }
QT_END_NAMESPACE

/**
 * @struct ArchiveFileInfo
 * @brief Information about a file in an archive
 */
struct ArchiveFileInfo {
    QString path;                  ///< Full path within archive
    QString name;                  ///< File name only
    uint64_t size;                 ///< Uncompressed size in bytes
    uint64_t compressedSize;       ///< Compressed size in bytes
    double compressionRatio;       ///< Ratio (compressed/uncompressed * 100)
    bool isDirectory;              ///< True if this is a directory entry
    QTreeWidgetItem* treeItem;     ///< Associated tree widget item
};

/**
 * @class ArchiveLoaderWorker
 * @brief Background worker thread for loading archive contents
 */
class ArchiveLoaderWorker : public QThread
{
    Q_OBJECT

public:
    explicit ArchiveLoaderWorker(const QString& archivePath, QObject* parent = nullptr);
    ~ArchiveLoaderWorker();

signals:
    void loadingProgress(int percentage, const QString& status);
    void loadingComplete(const QList<ArchiveFileInfo>& files, 
                        uint64_t totalSize, 
                        uint64_t totalCompressed,
                        int volumeCount);
    void loadingError(const QString& errorMessage);

protected:
    void run() override;

private:
    QString m_archivePath;
    bool m_canceled;
    
    /**
     * @brief Loads a compressed archive by decompressing it first
     * @param archivePath Path to the compressed archive
     * @param files Output list of files
     * @param totalSize Output total uncompressed size
     * @param totalCompressed Output total compressed size
     * @return true on success, false on error (emits loadingError)
     */
    bool loadCompressedArchive(const QString& archivePath, 
                              QList<ArchiveFileInfo>& files,
                              uint64_t& totalSize,
                              uint64_t& totalCompressed);
};

/**
 * @class ArchiveViewerDialog
 * @brief Dialog for viewing and extracting archive contents
 * 
 * Displays archive contents in a hierarchical tree view with file information,
 * supports searching/filtering, and provides extraction capabilities.
 */
class ArchiveViewerDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * @brief Constructs the archive viewer dialog
     * @param archivePath Path to the archive file to view
     * @param parent Parent widget (nullptr for top-level)
     */
    explicit ArchiveViewerDialog(const QString& archivePath, QWidget *parent = nullptr);
    
    /**
     * @brief Destroys the dialog and cleans up resources
     */
    ~ArchiveViewerDialog();

private slots:
    /**
     * @brief Handles Extract All button click
     */
    void onExtractAllClicked();
    
    /**
     * @brief Handles Extract Selected button click
     */
    void onExtractSelectedClicked();
    
    /**
     * @brief Handles Refresh button click
     */
    void onRefreshClicked();
    
    /**
     * @brief Handles Close button click
     */
    void onCloseClicked();
    
    /**
     * @brief Handles search text changes
     * @param text Search query text
     */
    void onSearchTextChanged(const QString& text);
    
    /**
     * @brief Handles tree item double-click
     * @param item The item that was double-clicked
     * @param column The column that was double-clicked
     */
    void onTreeItemDoubleClicked(QTreeWidgetItem* item, int column);
    
    /**
     * @brief Handles tree item selection changes
     */
    void onTreeSelectionChanged();
    
    /**
     * @brief Shows context menu for tree items
     * @param pos Position where menu should appear
     */
    void onTreeContextMenuRequested(const QPoint& pos);
    
    /**
     * @brief Handles loading progress updates
     * @param percentage Progress percentage (0-100)
     * @param status Status message
     */
    void onLoadingProgress(int percentage, const QString& status);
    
    /**
     * @brief Handles loading completion
     * @param files List of files in the archive
     * @param totalSize Total uncompressed size
     * @param totalCompressed Total compressed size
     * @param volumeCount Number of volumes
     */
    void onLoadingComplete(const QList<ArchiveFileInfo>& files,
                          uint64_t totalSize,
                          uint64_t totalCompressed,
                          int volumeCount);
    
    /**
     * @brief Handles loading errors
     * @param errorMessage Error description
     */
    void onLoadingError(const QString& errorMessage);
    
    /**
     * @brief Handles extract context menu action
     */
    void onExtractAction();
    
    /**
     * @brief Handles properties context menu action
     */
    void onPropertiesAction();

private:
    Ui::ArchiveViewerDialog *ui;  ///< Qt Designer generated UI
    QString m_archivePath;        ///< Path to the archive file
    QList<ArchiveFileInfo> m_files;  ///< List of files in archive
    QMap<QString, QTreeWidgetItem*> m_folderItems;  ///< Map of folder paths to tree items
    uint64_t m_totalSize;         ///< Total uncompressed size
    uint64_t m_totalCompressed;   ///< Total compressed size
    int m_volumeCount;            ///< Number of volumes in archive
    ArchiveLoaderWorker* m_loader;  ///< Background loader thread
    
    /**
     * @brief Initializes the UI components
     */
    void setupUi();
    
    /**
     * @brief Connects signals and slots
     */
    void setupConnections();
    
    /**
     * @brief Loads the archive contents
     */
    void loadArchive();
    
    /**
     * @brief Populates the tree view with archive contents
     */
    void populateTree();
    
    /**
     * @brief Creates or retrieves a folder item in the tree
     * @param folderPath Path to the folder
     * @return Tree widget item for the folder
     */
    QTreeWidgetItem* getOrCreateFolderItem(const QString& folderPath);
    
    /**
     * @brief Updates the statistics display
     */
    void updateStatistics();
    
    /**
     * @brief Filters tree items based on search query
     * @param query Search query string
     */
    void filterTreeItems(const QString& query);
    
    /**
     * @brief Sets visibility of tree item based on filter
     * @param item Tree item to update
     * @param query Search query
     * @return true if item matches query
     */
    bool filterTreeItem(QTreeWidgetItem* item, const QString& query);
    
    /**
     * @brief Extracts files from the archive
     * @param outputPath Destination directory
     * @param selectedOnly Extract only selected files
     */
    void extractFiles(const QString& outputPath, bool selectedOnly);
    
    /**
     * @brief Gets list of selected file paths
     * @return List of file paths to extract
     */
    QStringList getSelectedFilePaths();
    
    /**
     * @brief Formats a byte size in human-readable form
     * @param bytes Size in bytes
     * @return Formatted size string
     */
    QString formatSize(uint64_t bytes) const;
    
    /**
     * @brief Expands all parent items of a given item
     * @param item Item whose parents should be expanded
     */
    void expandParents(QTreeWidgetItem* item);
};

#endif // ARCHIVE_VIEWER_H

