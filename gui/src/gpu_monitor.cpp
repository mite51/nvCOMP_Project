/**
 * @file gpu_monitor.cpp
 * @brief Implementation of GPUMonitorWidget class
 */

#include "gpu_monitor.h"
#include "nvcomp_c_api.h"
#include <QGroupBox>
#include <QMessageBox>
#include <cuda_runtime.h>
#include <cmath>

GPUMonitorWidget::GPUMonitorWidget(QWidget *parent)
    : QWidget(parent)
    , m_refreshTimer(nullptr)
    , m_gpuCount(0)
    , m_tabWidget(nullptr)
    , m_refreshButton(nullptr)
{
    // Query GPU count
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess && deviceCount > 0) {
        m_gpuCount = deviceCount;
        m_gpuInfoCache.resize(deviceCount);
    }
    
    setupUI();
    
    // Initialize with current GPU info
    refresh();
    
    // Create timer for auto-refresh
    m_refreshTimer = new QTimer(this);
    connect(m_refreshTimer, &QTimer::timeout, this, &GPUMonitorWidget::onRefreshTimer);
    
    // Start auto-refresh at 500ms intervals
    startAutoRefresh();
}

GPUMonitorWidget::~GPUMonitorWidget()
{
    if (m_refreshTimer) {
        m_refreshTimer->stop();
    }
}

void GPUMonitorWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Title
    QLabel *titleLabel = new QLabel("<h2>GPU Monitor</h2>", this);
    titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(titleLabel);
    
    if (m_gpuCount == 0) {
        // No GPU available
        QLabel *noGPULabel = new QLabel("⚠️ No CUDA-compatible GPU detected", this);
        noGPULabel->setAlignment(Qt::AlignCenter);
        noGPULabel->setStyleSheet("font-size: 14pt; padding: 20px;");
        mainLayout->addWidget(noGPULabel);
        
        QLabel *infoLabel = new QLabel(
            "GPU monitoring requires a CUDA-compatible NVIDIA GPU.\n"
            "The application will run in CPU-only mode.",
            this
        );
        infoLabel->setAlignment(Qt::AlignCenter);
        infoLabel->setWordWrap(true);
        mainLayout->addWidget(infoLabel);
    } else if (m_gpuCount == 1) {
        // Single GPU - no tabs needed
        QWidget *gpuWidget = createGPUTab(0);
        mainLayout->addWidget(gpuWidget);
    } else {
        // Multiple GPUs - use tabs
        m_tabWidget = new QTabWidget(this);
        for (int i = 0; i < m_gpuCount; ++i) {
            QWidget *gpuWidget = createGPUTab(i);
            m_gpuTabs.push_back(gpuWidget);
            m_tabWidget->addTab(gpuWidget, QString("GPU %1").arg(i));
        }
        mainLayout->addWidget(m_tabWidget);
    }
    
    // Refresh button
    m_refreshButton = new QPushButton("🔄 Refresh Now", this);
    connect(m_refreshButton, &QPushButton::clicked, this, &GPUMonitorWidget::onRefreshButtonClicked);
    mainLayout->addWidget(m_refreshButton);
    
    // Stretch to push everything up
    mainLayout->addStretch();
    
    setLayout(mainLayout);
}

QWidget* GPUMonitorWidget::createGPUTab(int deviceIndex)
{
    QWidget *widget = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(widget);
    
    // GPU Info Group
    QGroupBox *infoGroup = new QGroupBox("GPU Information", widget);
    QGridLayout *infoLayout = new QGridLayout(infoGroup);
    
    // GPU Name
    QLabel *nameLabel = new QLabel("<b>GPU Name:</b>", infoGroup);
    QLabel *nameValue = new QLabel("Loading...", infoGroup);
    infoLayout->addWidget(nameLabel, 0, 0);
    infoLayout->addWidget(nameValue, 0, 1);
    m_nameLabels.push_back(nameValue);
    
    // CUDA Version
    QLabel *cudaLabel = new QLabel("<b>CUDA Version:</b>", infoGroup);
    QLabel *cudaValue = new QLabel("Loading...", infoGroup);
    infoLayout->addWidget(cudaLabel, 1, 0);
    infoLayout->addWidget(cudaValue, 1, 1);
    m_cudaVersionLabels.push_back(cudaValue);
    
    // Temperature (if available)
    QLabel *tempLabel = new QLabel("<b>Temperature:</b>", infoGroup);
    QLabel *tempValue = new QLabel("N/A", infoGroup);
    infoLayout->addWidget(tempLabel, 2, 0);
    infoLayout->addWidget(tempValue, 2, 1);
    m_temperatureLabels.push_back(tempValue);
    
    layout->addWidget(infoGroup);
    
    // VRAM Group
    QGroupBox *vramGroup = new QGroupBox("VRAM Usage", widget);
    QVBoxLayout *vramLayout = new QVBoxLayout(vramGroup);
    
    QGridLayout *vramInfoLayout = new QGridLayout();
    
    // Total VRAM
    QLabel *totalLabel = new QLabel("<b>Total:</b>", vramGroup);
    QLabel *totalValue = new QLabel("0 GB", vramGroup);
    vramInfoLayout->addWidget(totalLabel, 0, 0);
    vramInfoLayout->addWidget(totalValue, 0, 1);
    m_totalVRAMLabels.push_back(totalValue);
    
    // Free VRAM
    QLabel *freeLabel = new QLabel("<b>Free:</b>", vramGroup);
    QLabel *freeValue = new QLabel("0 GB", vramGroup);
    vramInfoLayout->addWidget(freeLabel, 1, 0);
    vramInfoLayout->addWidget(freeValue, 1, 1);
    m_freeVRAMLabels.push_back(freeValue);
    
    // Used VRAM
    QLabel *usedLabel = new QLabel("<b>Used:</b>", vramGroup);
    QLabel *usedValue = new QLabel("0 GB", vramGroup);
    vramInfoLayout->addWidget(usedLabel, 2, 0);
    vramInfoLayout->addWidget(usedValue, 2, 1);
    m_usedVRAMLabels.push_back(usedValue);
    
    vramLayout->addLayout(vramInfoLayout);
    
    // Progress bar
    QProgressBar *vramBar = new QProgressBar(vramGroup);
    vramBar->setRange(0, 100);
    vramBar->setValue(0);
    vramBar->setTextVisible(true);
    vramBar->setFormat("%p% Used");
    vramLayout->addWidget(vramBar);
    m_vramProgressBars.push_back(vramBar);
    
    // Warning label
    QLabel *warningLabel = new QLabel("", vramGroup);
    warningLabel->setWordWrap(true);
    warningLabel->setAlignment(Qt::AlignCenter);
    warningLabel->setStyleSheet("font-weight: bold; padding: 5px;");
    vramLayout->addWidget(warningLabel);
    m_warningLabels.push_back(warningLabel);
    
    layout->addWidget(vramGroup);
    
    // Stretch to push content up
    layout->addStretch();
    
    widget->setLayout(layout);
    return widget;
}

GPUInfo GPUMonitorWidget::queryGPUInfo(int deviceIndex) const
{
    GPUInfo info;
    info.deviceIndex = deviceIndex;
    info.available = false;
    info.temperature = -1;  // Not available without NVML
    
    cudaError_t err = cudaSetDevice(deviceIndex);
    if (err != cudaSuccess) {
        return info;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceIndex);
    if (err != cudaSuccess) {
        return info;
    }
    
    info.name = prop.name;
    info.cudaVersionMajor = prop.major;
    info.cudaVersionMinor = prop.minor;
    info.totalVRAM = prop.totalGlobalMem;
    info.available = true;
    
    // Get memory info
    size_t free, total;
    err = cudaMemGetInfo(&free, &total);
    if (err == cudaSuccess) {
        info.freeVRAM = free;
        info.usedVRAM = total - free;
    } else {
        info.freeVRAM = 0;
        info.usedVRAM = 0;
    }
    
    // Note: Temperature requires NVML (NVIDIA Management Library)
    // We skip it for now to keep dependencies simple
    // Future enhancement: Add NVML support for temperature
    
    return info;
}

void GPUMonitorWidget::updateGPUDisplay(int deviceIndex)
{
    if (deviceIndex < 0 || deviceIndex >= m_gpuCount) {
        return;
    }
    
    GPUInfo info = queryGPUInfo(deviceIndex);
    m_gpuInfoCache[deviceIndex] = info;
    
    if (!info.available) {
        m_nameLabels[deviceIndex]->setText("GPU not available");
        return;
    }
    
    // Update name
    m_nameLabels[deviceIndex]->setText(QString::fromStdString(info.name));
    
    // Update CUDA version
    m_cudaVersionLabels[deviceIndex]->setText(
        QString("Compute %1.%2").arg(info.cudaVersionMajor).arg(info.cudaVersionMinor)
    );
    
    // Update temperature
    if (info.temperature >= 0) {
        m_temperatureLabels[deviceIndex]->setText(QString("%1°C").arg(info.temperature));
    } else {
        m_temperatureLabels[deviceIndex]->setText("N/A");
    }
    
    // Update VRAM info
    m_totalVRAMLabels[deviceIndex]->setText(formatBytes(info.totalVRAM));
    m_freeVRAMLabels[deviceIndex]->setText(formatBytes(info.freeVRAM));
    m_usedVRAMLabels[deviceIndex]->setText(formatBytes(info.usedVRAM));
    
    // Update progress bar
    float percentUsed = 0.0f;
    if (info.totalVRAM > 0) {
        percentUsed = (static_cast<float>(info.usedVRAM) / static_cast<float>(info.totalVRAM)) * 100.0f;
    }
    
    m_vramProgressBars[deviceIndex]->setValue(static_cast<int>(percentUsed));
    
    // Color-code progress bar
    QString color = getColorForUsage(percentUsed);
    m_vramProgressBars[deviceIndex]->setStyleSheet(
        QString("QProgressBar::chunk { background-color: %1; }").arg(color)
    );
    
    // Update warning label
    float percentFree = 100.0f - percentUsed;
    if (percentFree < 10.0f) {
        m_warningLabels[deviceIndex]->setText("⚠️ WARNING: Low VRAM available!");
        m_warningLabels[deviceIndex]->setStyleSheet(
            "background-color: #ffcccc; color: #cc0000; font-weight: bold; padding: 5px; border-radius: 3px;"
        );
        emit vramLowWarning(deviceIndex, percentFree);
    } else if (percentFree < 25.0f) {
        m_warningLabels[deviceIndex]->setText("⚠️ Caution: VRAM usage is high");
        m_warningLabels[deviceIndex]->setStyleSheet(
            "background-color: #fff3cd; color: #856404; font-weight: bold; padding: 5px; border-radius: 3px;"
        );
    } else {
        m_warningLabels[deviceIndex]->setText("✅ VRAM OK");
        m_warningLabels[deviceIndex]->setStyleSheet(
            "background-color: #d4edda; color: #155724; font-weight: bold; padding: 5px; border-radius: 3px;"
        );
    }
    
    emit gpuInfoUpdated(deviceIndex, info);
}

void GPUMonitorWidget::refresh()
{
    for (int i = 0; i < m_gpuCount; ++i) {
        updateGPUDisplay(i);
    }
}

void GPUMonitorWidget::setRefreshInterval(int intervalMs)
{
    if (m_refreshTimer) {
        m_refreshTimer->setInterval(intervalMs);
    }
}

void GPUMonitorWidget::startAutoRefresh()
{
    if (m_refreshTimer && !m_refreshTimer->isActive()) {
        m_refreshTimer->start(500);  // 500ms default
    }
}

void GPUMonitorWidget::stopAutoRefresh()
{
    if (m_refreshTimer) {
        m_refreshTimer->stop();
    }
}

void GPUMonitorWidget::onRefreshTimer()
{
    refresh();
}

void GPUMonitorWidget::onRefreshButtonClicked()
{
    refresh();
}

GPUInfo GPUMonitorWidget::getGPUInfo(int deviceIndex) const
{
    if (deviceIndex >= 0 && deviceIndex < static_cast<int>(m_gpuInfoCache.size())) {
        return m_gpuInfoCache[deviceIndex];
    }
    GPUInfo info;
    info.available = false;
    return info;
}

bool GPUMonitorWidget::isGPUAvailable() const
{
    return m_gpuCount > 0;
}

int GPUMonitorWidget::getGPUCount() const
{
    return m_gpuCount;
}

size_t GPUMonitorWidget::predictVRAMNeeded(uint64_t fileSize) const
{
    // Conservative estimate: 2x file size for compression workspace
    // (input buffer + output buffer + temporary workspace)
    return static_cast<size_t>(fileSize * 2.5);
}

bool GPUMonitorWidget::checkVRAMSufficient(size_t requiredVRAM, int deviceIndex) const
{
    if (deviceIndex < 0 || deviceIndex >= static_cast<int>(m_gpuInfoCache.size())) {
        return false;
    }
    
    const GPUInfo &info = m_gpuInfoCache[deviceIndex];
    if (!info.available) {
        return false;
    }
    
    // Check if free VRAM is at least the required amount plus 10% safety margin
    size_t safetyMargin = static_cast<size_t>(info.totalVRAM * 0.1);
    return info.freeVRAM >= (requiredVRAM + safetyMargin);
}

QString GPUMonitorWidget::formatBytes(size_t bytes) const
{
    if (bytes == 0) {
        return "0 B";
    }
    
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }
    
    return QString("%1 %2").arg(size, 0, 'f', 2).arg(units[unitIndex]);
}

QString GPUMonitorWidget::getColorForUsage(float percentUsed) const
{
    if (percentUsed >= 90.0f) {
        return "#cc0000";  // Red - critical
    } else if (percentUsed >= 75.0f) {
        return "#ff9900";  // Orange - high
    } else if (percentUsed >= 50.0f) {
        return "#ffcc00";  // Yellow - moderate
    } else {
        return "#00cc00";  // Green - good
    }
}

