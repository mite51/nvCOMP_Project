/**
 * @file gpu_monitor.h
 * @brief GPU monitoring widget for displaying real-time GPU status
 * 
 * Provides GPU information including VRAM usage, temperature, and CUDA capabilities
 */

#ifndef GPU_MONITOR_H
#define GPU_MONITOR_H

#include <QWidget>
#include <QTimer>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QGridLayout>
#include <vector>

/**
 * @struct GPUInfo
 * @brief Structure holding GPU information
 */
struct GPUInfo {
    int deviceIndex;
    std::string name;
    int cudaVersionMajor;
    int cudaVersionMinor;
    size_t totalVRAM;
    size_t freeVRAM;
    size_t usedVRAM;
    int temperature;  // -1 if unavailable
    bool available;
};

/**
 * @class GPUMonitorWidget
 * @brief Widget for monitoring GPU status in real-time
 * 
 * Displays GPU information including:
 * - GPU name and model
 * - CUDA driver version
 * - Total/Free/Used VRAM
 * - VRAM usage percentage
 * - GPU temperature (if NVML available)
 * - Color-coded warnings for low VRAM
 */
class GPUMonitorWidget : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief Constructs the GPU monitor widget
     * @param parent Parent widget (nullptr for standalone window)
     */
    explicit GPUMonitorWidget(QWidget *parent = nullptr);
    
    /**
     * @brief Destroys the GPU monitor widget
     */
    ~GPUMonitorWidget();
    
    /**
     * @brief Gets the current GPU information
     * @param deviceIndex GPU device index (default 0)
     * @return GPUInfo structure
     */
    GPUInfo getGPUInfo(int deviceIndex = 0) const;
    
    /**
     * @brief Checks if any GPU is available
     * @return true if at least one GPU is available
     */
    bool isGPUAvailable() const;
    
    /**
     * @brief Gets the number of available GPUs
     * @return Number of GPUs
     */
    int getGPUCount() const;
    
    /**
     * @brief Predicts VRAM needed for a given file size
     * @param fileSize File size in bytes
     * @return Estimated VRAM needed in bytes
     */
    size_t predictVRAMNeeded(uint64_t fileSize) const;
    
    /**
     * @brief Checks if GPU has sufficient VRAM for operation
     * @param requiredVRAM Required VRAM in bytes
     * @param deviceIndex GPU device index (default 0)
     * @return true if sufficient VRAM available
     */
    bool checkVRAMSufficient(size_t requiredVRAM, int deviceIndex = 0) const;

signals:
    /**
     * @brief Emitted when VRAM is low (<10% free)
     * @param deviceIndex GPU device index
     * @param percentFree Percentage of free VRAM
     */
    void vramLowWarning(int deviceIndex, float percentFree);
    
    /**
     * @brief Emitted when GPU information is updated
     * @param deviceIndex GPU device index
     * @param info Updated GPU information
     */
    void gpuInfoUpdated(int deviceIndex, const GPUInfo &info);

public slots:
    /**
     * @brief Refreshes GPU information immediately
     */
    void refresh();
    
    /**
     * @brief Sets the auto-refresh interval
     * @param intervalMs Interval in milliseconds (500ms default)
     */
    void setRefreshInterval(int intervalMs);
    
    /**
     * @brief Starts auto-refresh timer
     */
    void startAutoRefresh();
    
    /**
     * @brief Stops auto-refresh timer
     */
    void stopAutoRefresh();

private slots:
    /**
     * @brief Timer callback for auto-refresh
     */
    void onRefreshTimer();
    
    /**
     * @brief Handles refresh button click
     */
    void onRefreshButtonClicked();

private:
    QTimer *m_refreshTimer;
    int m_gpuCount;
    std::vector<GPUInfo> m_gpuInfoCache;
    
    // UI components
    QTabWidget *m_tabWidget;  // For multiple GPUs
    std::vector<QWidget*> m_gpuTabs;
    std::vector<QLabel*> m_nameLabels;
    std::vector<QLabel*> m_cudaVersionLabels;
    std::vector<QLabel*> m_totalVRAMLabels;
    std::vector<QLabel*> m_freeVRAMLabels;
    std::vector<QLabel*> m_usedVRAMLabels;
    std::vector<QLabel*> m_temperatureLabels;
    std::vector<QProgressBar*> m_vramProgressBars;
    std::vector<QLabel*> m_warningLabels;
    QPushButton *m_refreshButton;
    
    /**
     * @brief Initializes the UI components
     */
    void setupUI();
    
    /**
     * @brief Creates a single GPU tab
     * @param deviceIndex GPU device index
     * @return Widget containing GPU information display
     */
    QWidget* createGPUTab(int deviceIndex);
    
    /**
     * @brief Updates UI with current GPU information
     * @param deviceIndex GPU device index
     */
    void updateGPUDisplay(int deviceIndex);
    
    /**
     * @brief Queries GPU information from CUDA
     * @param deviceIndex GPU device index
     * @return GPUInfo structure
     */
    GPUInfo queryGPUInfo(int deviceIndex) const;
    
    /**
     * @brief Formats bytes to human-readable string
     * @param bytes Number of bytes
     * @return Formatted string (e.g., "4.5 GB")
     */
    QString formatBytes(size_t bytes) const;
    
    /**
     * @brief Gets color based on usage percentage
     * @param percentUsed Percentage of VRAM used (0-100)
     * @return Color string for styling
     */
    QString getColorForUsage(float percentUsed) const;
};

#endif // GPU_MONITOR_H

