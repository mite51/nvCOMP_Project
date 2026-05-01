/**
 * @file mainwindow.h
 * @brief Main window class for nvCOMP GUI application
 * 
 * Provides the primary user interface for compression/decompression operations.
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QStringList>

#include "nvcomp_c_api.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class CompressionWorker;
class QListWidgetItem;
class SettingsDialog;
class GPUMonitorWidget;
class ProgressWidget;

/**
 * @class MainWindow
 * @brief Main application window
 * 
 * Handles user interactions, file operations, and displays compression progress.
 * Supports drag-and-drop, multiple compression algorithms, and GPU/CPU modes.
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    /**
     * @brief Constructs the main window
     * @param parent Parent widget (nullptr for top-level window)
     */
    explicit MainWindow(QWidget *parent = nullptr);
    
    /**
     * @brief Destroys the main window and cleans up resources
     */
    ~MainWindow();
    
    /**
     * @brief Gets the window title
     * @return Current window title string
     */
    QString windowTitle() const;
    
    /**
     * @brief Checks if the main window is properly initialized
     * @return true if window is valid and ready
     */
    bool isInitialized() const;
    
    /**
     * @brief Gets the list of files currently in the file list
     * @return QStringList of file paths
     */
    QStringList getFileList() const;
    
    /**
     * @brief Gets the currently selected algorithm
     * @return QString containing the selected algorithm name
     */
    QString getSelectedAlgorithm() const;
    
    /**
     * @brief Checks if CPU mode is enabled
     * @return true if CPU mode is forced
     */
    bool isCpuModeEnabled() const;
    
    /**
     * @brief Checks if volume splitting is enabled
     * @return true if volumes are enabled
     */
    bool isVolumesEnabled() const;
    
    /**
     * @brief Gets the volume size setting
     * @return Volume size in MB
     */
    int getVolumeSize() const;
    
    /**
     * @brief Gets the output archive name
     * @return Output archive filename (may be empty for auto-generated)
     */
    QString getOutputArchiveName() const;
    
    /**
     * @brief Adds files from command-line arguments
     * @param files List of file paths to add
     * 
     * Used when application is launched from context menu or command line
     */
    void addFilesFromCommandLine(const QStringList &files);
    
    /**
     * @brief Sets the algorithm from command-line argument
     * @param algorithm Algorithm name (lz4, snappy, zstd, etc.)
     * 
     * Selects the appropriate algorithm in the combo box
     */
    void setAlgorithmFromCommandLine(const QString &algorithm);
    
    /**
     * @brief Starts compression automatically (for context menu)
     * 
     * Used when --compress flag is passed on command line
     */
    void startCompressionFromCommandLine();
    
    /**
     * @brief Starts decompression automatically (for context menu)
     * @param outputDir Optional output directory (empty for dialog)
     * 
     * Used when --decompress flag is passed on command line
     */
    void startDecompressionFromCommandLine(const QString &outputDir = QString());

protected:
    /**
     * @brief Handles drag enter events for file drag-and-drop
     * @param event Drag enter event
     */
    void dragEnterEvent(QDragEnterEvent *event) override;
    
    /**
     * @brief Handles drop events for file drag-and-drop
     * @param event Drop event
     */
    void dropEvent(QDropEvent *event) override;

private:
    Ui::MainWindow *ui;  ///< Qt Designer generated UI
    bool m_initialized;  ///< Initialization state flag
    QStringList m_fileList;  ///< List of files to process
    bool m_gpuAvailable;  ///< GPU availability status
    CompressionWorker *m_worker;  ///< Background compression worker thread
    SettingsDialog *m_settingsDialog;  ///< Settings dialog (created on demand)
    GPUMonitorWidget *m_gpuMonitor;  ///< GPU monitor widget (created on demand)
    ProgressWidget *m_progressWidget;  ///< Advanced progress widget (embedded or hidden)
    QString m_currentStage;  ///< Current processing stage (reading, compressing, writing)
    
    /**
     * @brief Initializes UI components and connections
     */
    void setupUi();
    
    /**
     * @brief Connects signals and slots for UI interactions
     */
    void setupConnections();
    
    /**
     * @brief Checks GPU availability and updates status
     */
    void checkGpuAvailability();
    
    /**
     * @brief Updates UI state based on file list
     */
    void updateUiState();
    
    /**
     * @brief Adds files and folders to the list
     * @param paths List of file or folder paths to add
     */
    void addFiles(const QStringList &paths);
    
    /**
     * @brief Applies settings from the settings dialog to the UI
     */
    void applySettingsToUi();
    
    /**
     * @brief Applies the selected theme (Light/Dark/System)
     * @param theme Theme name: "Light", "Dark", or "System"
     */
    void applyTheme(const QString &theme);
    
private slots:
    /**
     * @brief Handles About dialog display
     * Shows application information and version
     */
    void onAboutTriggered();
    
    /**
     * @brief Handles application exit
     * Performs cleanup before closing
     */
    void onExitTriggered();
    
    /**
     * @brief Handles Add Files button click
     * Opens dialog to select files or folders
     */
    void onAddFilesClicked();
    
    /**
     * @brief Deprecated: Forwards to onAddFilesClicked
     */
    void onAddFolderClicked();
    
    /**
     * @brief Handles Clear Selected button click
     * Removes selected files from the list
     */
    void onClearSelectedClicked();
    
    /**
     * @brief Handles Clear Files button click
     * Removes all files from the list
     */
    void onClearFilesClicked();
    
    /**
     * @brief Handles Compress button click
     * Initiates compression operation (stub for now)
     */
    void onCompressClicked();
    
    /**
     * @brief Handles Decompress button click
     * Initiates decompression operation (stub for now)
     */
    void onDecompressClicked();
    
    /**
     * @brief Handles Settings button click
     * Opens settings dialog (stub for now)
     */
    void onSettingsClicked();
    
    /**
     * @brief Handles algorithm selection change
     * @param index Index of selected algorithm
     */
    void onAlgorithmChanged(int index);
    
    /**
     * @brief Handles CPU mode checkbox toggle
     * @param checked True if CPU mode is enabled
     */
    void onCpuModeToggled(bool checked);
    
    /**
     * @brief Handles Browse button click for output archive selection
     * Opens file dialog to select output location and filename
     */
    void onBrowseOutputClicked();
    
    /**
     * @brief Single throttled progress update from the worker.
     *
     * Replaces the previous fan-out of 7 separate per-chunk signals
     * (totalBlocksChanged / blockProgressChanged / blockCompleted /
     * throughputChanged / stageChanged / progressChanged / progressDetails),
     * which were the main reason the GUI was ~2x slower than the CLI.
     */
    void onProgressUpdate(int percent, const QString &stage, double mbps, qint64 elapsedMs);
    
    /**
     * @brief Handles worker completion. Receives the per-phase stats produced
     *        by the core so the GUI renders the same summary as the CLI.
     */
    void onWorkerFinished(const QString &outputPath, const nvcomp_compression_stats_t &stats);
    
    /**
     * @brief Handles worker errors
     * @param errorMessage Error description
     */
    void onWorkerError(const QString &errorMessage);
    
    /**
     * @brief Handles worker cancellation
     */
    void onWorkerCanceled();
    
    /**
     * @brief Handles worker status messages
     * @param message Status message for display
     */
    void onWorkerStatusMessage(const QString &message);
    
    /**
     * @brief Handles View Archive menu action
     * Opens archive viewer dialog for a selected archive file
     */
    void onViewArchiveTriggered();
    
    /**
     * @brief Handles double-click on file list items
     * Opens archive viewer if the file is a compressed archive
     * @param item The item that was double-clicked
     */
    void onFileListDoubleClicked(QListWidgetItem* item);
    
    /**
     * @brief Handles GPU Monitor menu action
     * Opens GPU monitoring window
     */
    void onGPUMonitorTriggered();
    
    /**
     * @brief Handles VRAM low warning from GPU monitor
     * @param deviceIndex GPU device index
     * @param percentFree Percent free VRAM
     */
    void onVRAMLowWarning(int deviceIndex, float percentFree);
    
    /**
     * @brief Handles context menu registration (Windows only)
     * Registers Windows Explorer context menu integration
     */
    void onRegisterContextMenu();
    
    /**
     * @brief Handles context menu unregistration (Windows only)
     * Removes Windows Explorer context menu integration
     */
    void onUnregisterContextMenu();
    
    /**
     * @brief Handles file association registration (Windows only)
     * Registers file associations for compressed archive types
     */
    void onRegisterFileAssociations();
    
    /**
     * @brief Handles file association unregistration (Windows only)
     * Removes file associations for compressed archive types
     */
    void onUnregisterFileAssociations();
};

#endif // MAINWINDOW_H


