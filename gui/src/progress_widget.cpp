/**
 * @file progress_widget.cpp
 * @brief Implementation of ProgressWidget class
 */

#include "progress_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QPaintEvent>
#include <QToolTip>
#include <QMouseEvent>
#include <cmath>

ProgressWidget::ProgressWidget(QWidget *parent)
    : QWidget(parent)
    , m_totalBlocks(0)
    , m_completedBlocks(0)
    , m_currentBlock(-1)
    , m_overallProgress(0.0f)
    , m_throughput(0.0)
    , m_currentBytes(0)
    , m_totalBytes(0)
    , m_etaSeconds(0)
    , m_titleLabel(nullptr)
    , m_overallProgressBar(nullptr)
    , m_stageLabel(nullptr)
    , m_fileLabel(nullptr)
    , m_dataLabel(nullptr)
    , m_speedLabel(nullptr)
    , m_etaLabel(nullptr)
    , m_blockInfoLabel(nullptr)
    , m_blockGridWidget(nullptr)
    , m_updateTimer(nullptr)
    , m_needsRepaint(false)
    , m_blockGridRows(0)
    , m_blockGridCols(0)
    , m_maxDisplayBlocks(50)
    , m_blocksPerAggregate(1)
{
    setupUI();
    
    // Create update timer for throttled repaints (30 FPS)
    m_updateTimer = new QTimer(this);
    m_updateTimer->setInterval(33);  // ~30 FPS
    connect(m_updateTimer, &QTimer::timeout, this, &ProgressWidget::onUpdateTimer);
    m_updateTimer->start();
    
    // Enable mouse tracking for tooltips
    setMouseTracking(true);
}

ProgressWidget::~ProgressWidget()
{
    if (m_updateTimer) {
        m_updateTimer->stop();
    }
}

void ProgressWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(10);
    
    // Title
    m_titleLabel = new QLabel("<h3>Compression Progress</h3>", this);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_titleLabel);
    
    // Overall progress bar
    m_overallProgressBar = new QProgressBar(this);
    m_overallProgressBar->setRange(0, 100);
    m_overallProgressBar->setValue(0);
    m_overallProgressBar->setTextVisible(true);
    m_overallProgressBar->setFormat("%p%");
    m_overallProgressBar->setMinimumHeight(30);
    mainLayout->addWidget(m_overallProgressBar);
    
    // Current file and stage
    QHBoxLayout *fileStageLayout = new QHBoxLayout();
    m_fileLabel = new QLabel("File: (none)", this);
    m_stageLabel = new QLabel("Stage: Idle", this);
    fileStageLayout->addWidget(m_fileLabel, 1);
    fileStageLayout->addWidget(m_stageLabel, 1);
    mainLayout->addLayout(fileStageLayout);
    
    // Stats group
    QGroupBox *statsGroup = new QGroupBox("Statistics", this);
    QGridLayout *statsLayout = new QGridLayout(statsGroup);
    
    m_dataLabel = new QLabel("Data: 0 B / 0 B", this);
    m_speedLabel = new QLabel("Speed: 0.00 MB/s", this);
    m_etaLabel = new QLabel("ETA: --", this);
    
    statsLayout->addWidget(m_dataLabel, 0, 0);
    statsLayout->addWidget(m_speedLabel, 0, 1);
    statsLayout->addWidget(m_etaLabel, 0, 2);
    
    mainLayout->addWidget(statsGroup);
    
    // Block grid visualization
    QGroupBox *blockGroup = new QGroupBox("Block Progress", this);
    QVBoxLayout *blockLayout = new QVBoxLayout(blockGroup);
    
    m_blockInfoLabel = new QLabel("Blocks: 0 / 0 complete", this);
    m_blockInfoLabel->setAlignment(Qt::AlignCenter);
    blockLayout->addWidget(m_blockInfoLabel);
    
    // Custom widget for block grid
    // We'll create a custom painted widget for the block grid
    class BlockGridWidget : public QWidget {
    public:
        ProgressWidget* parent;
        BlockGridWidget(ProgressWidget* p) : QWidget(p), parent(p) {}
    protected:
        void paintEvent(QPaintEvent* event) override {
            QPainter painter(this);
            painter.setRenderHint(QPainter::Antialiasing, false);
            parent->drawBlockGrid(painter, rect());
        }
    };
    
    m_blockGridWidget = new BlockGridWidget(this);
    m_blockGridWidget->setMinimumHeight(150);
    m_blockGridWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_blockGridWidget->setMouseTracking(true);
    blockLayout->addWidget(m_blockGridWidget);
    
    mainLayout->addWidget(blockGroup, 1);  // Give it stretch priority
    
    setLayout(mainLayout);
}

void ProgressWidget::setTotalBlocks(int total)
{
    m_totalBlocks = total;
    m_blocks.resize(total);
    m_completedBlocks = 0;
    m_currentBlock = -1;
    
    // Reset all blocks to pending
    for (int i = 0; i < total; ++i) {
        m_blocks[i].status = Pending;
        m_blocks[i].progress = 0.0f;
        m_blocks[i].compressionRatio = 0.0f;
    }
    
    calculateBlockGrid();
    m_blockInfoLabel->setText(QString("Blocks: 0 / %1 complete").arg(total));
    m_needsRepaint = true;
}

void ProgressWidget::updateBlockProgress(int blockIndex, float progress)
{
    if (blockIndex < 0 || blockIndex >= m_blocks.size()) {
        return;
    }
    
    m_blocks[blockIndex].status = Processing;
    m_blocks[blockIndex].progress = qBound(0.0f, progress, 1.0f);
    m_currentBlock = blockIndex;
    m_needsRepaint = true;
}

void ProgressWidget::setBlockComplete(int blockIndex, float compressionRatio)
{
    if (blockIndex < 0 || blockIndex >= m_blocks.size()) {
        return;
    }
    
    if (m_blocks[blockIndex].status != Complete) {
        m_completedBlocks++;
    }
    
    m_blocks[blockIndex].status = Complete;
    m_blocks[blockIndex].progress = 1.0f;
    m_blocks[blockIndex].compressionRatio = compressionRatio;
    
    m_blockInfoLabel->setText(QString("Blocks: %1 / %2 complete").arg(m_completedBlocks).arg(m_totalBlocks));
    m_needsRepaint = true;
}

void ProgressWidget::setBlockFailed(int blockIndex)
{
    if (blockIndex < 0 || blockIndex >= m_blocks.size()) {
        return;
    }
    
    m_blocks[blockIndex].status = Failed;
    m_blocks[blockIndex].progress = 0.0f;
    m_needsRepaint = true;
}

void ProgressWidget::updateOverallProgress(float progress)
{
    m_overallProgress = qBound(0.0f, progress, 1.0f);
    m_overallProgressBar->setValue(static_cast<int>(m_overallProgress * 100.0f));
}

void ProgressWidget::updateThroughput(double mbps)
{
    m_throughput = mbps;
    
    if (mbps >= 1000.0) {
        m_speedLabel->setText(QString("Speed: %1 GB/s").arg(mbps / 1024.0, 0, 'f', 2));
    } else {
        m_speedLabel->setText(QString("Speed: %1 MB/s").arg(mbps, 0, 'f', 2));
    }
}

void ProgressWidget::setCurrentStage(const QString &stage)
{
    m_currentStage = stage;
    m_stageLabel->setText(QString("Stage: %1").arg(stage));
}

void ProgressWidget::setCurrentFile(const QString &filename)
{
    m_currentFile = filename;
    m_fileLabel->setText(QString("File: %1").arg(filename));
}

void ProgressWidget::setDataProgress(uint64_t current, uint64_t total)
{
    m_currentBytes = current;
    m_totalBytes = total;
    m_dataLabel->setText(QString("Data: %1 / %2")
                            .arg(formatBytes(current))
                            .arg(formatBytes(total)));
}

void ProgressWidget::setETA(int seconds)
{
    m_etaSeconds = seconds;
    if (seconds > 0) {
        m_etaLabel->setText(QString("ETA: %1").arg(formatTime(seconds)));
    } else {
        m_etaLabel->setText("ETA: --");
    }
}

void ProgressWidget::reset()
{
    m_totalBlocks = 0;
    m_completedBlocks = 0;
    m_currentBlock = -1;
    m_overallProgress = 0.0f;
    m_throughput = 0.0;
    m_currentStage.clear();
    m_currentFile.clear();
    m_currentBytes = 0;
    m_totalBytes = 0;
    m_etaSeconds = 0;
    m_blocks.clear();
    
    m_overallProgressBar->setValue(0);
    m_stageLabel->setText("Stage: Idle");
    m_fileLabel->setText("File: (none)");
    m_dataLabel->setText("Data: 0 B / 0 B");
    m_speedLabel->setText("Speed: 0.00 MB/s");
    m_etaLabel->setText("ETA: --");
    m_blockInfoLabel->setText("Blocks: 0 / 0 complete");
    
    m_needsRepaint = true;
}

void ProgressWidget::calculateBlockGrid()
{
    if (m_totalBlocks == 0) {
        m_blockGridRows = 0;
        m_blockGridCols = 0;
        m_blocksPerAggregate = 1;
        return;
    }
    
    // Determine how many blocks to actually display
    int displayBlocks = m_totalBlocks;
    m_blocksPerAggregate = 1;
    
    if (m_totalBlocks > m_maxDisplayBlocks) {
        // Aggregate blocks for display
        m_blocksPerAggregate = (m_totalBlocks + m_maxDisplayBlocks - 1) / m_maxDisplayBlocks;
        displayBlocks = (m_totalBlocks + m_blocksPerAggregate - 1) / m_blocksPerAggregate;
    }
    
    // Calculate grid dimensions (try to keep roughly square)
    m_blockGridCols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(displayBlocks))));
    m_blockGridRows = (displayBlocks + m_blockGridCols - 1) / m_blockGridCols;
}

void ProgressWidget::paintEvent(QPaintEvent *event)
{
    // Default painting for the widget itself
    QWidget::paintEvent(event);
}

void ProgressWidget::drawBlockGrid(QPainter &painter, const QRect &rect)
{
    if (m_totalBlocks == 0 || m_blockGridRows == 0 || m_blockGridCols == 0) {
        // Draw "No blocks" message
        painter.setPen(Qt::gray);
        painter.drawText(rect, Qt::AlignCenter, "No blocks to display");
        return;
    }
    
    int padding = 4;
    int blockWidth = (rect.width() - padding * (m_blockGridCols + 1)) / m_blockGridCols;
    int blockHeight = (rect.height() - padding * (m_blockGridRows + 1)) / m_blockGridRows;
    
    // Ensure minimum size
    blockWidth = qMax(blockWidth, 10);
    blockHeight = qMax(blockHeight, 10);
    
    int displayBlocks = (m_totalBlocks + m_blocksPerAggregate - 1) / m_blocksPerAggregate;
    
    for (int i = 0; i < displayBlocks; ++i) {
        int row = i / m_blockGridCols;
        int col = i % m_blockGridCols;
        
        int x = padding + col * (blockWidth + padding);
        int y = padding + row * (blockHeight + padding);
        
        QRect blockRect(x, y, blockWidth, blockHeight);
        
        // Determine aggregate block status
        BlockState aggregateState;
        aggregateState.status = Pending;
        aggregateState.progress = 0.0f;
        aggregateState.compressionRatio = 0.0f;
        
        int startBlock = i * m_blocksPerAggregate;
        int endBlock = qMin(startBlock + m_blocksPerAggregate, m_totalBlocks);
        int blockCount = endBlock - startBlock;
        
        float totalProgress = 0.0f;
        float totalRatio = 0.0f;
        int completeCount = 0;
        int processingCount = 0;
        int failedCount = 0;
        
        for (int j = startBlock; j < endBlock; ++j) {
            const BlockState &block = m_blocks[j];
            totalProgress += block.progress;
            totalRatio += block.compressionRatio;
            
            if (block.status == Complete) completeCount++;
            else if (block.status == Processing) processingCount++;
            else if (block.status == Failed) failedCount++;
        }
        
        // Determine overall status
        if (completeCount == blockCount) {
            aggregateState.status = Complete;
            aggregateState.progress = 1.0f;
            aggregateState.compressionRatio = totalRatio / blockCount;
        } else if (failedCount > 0) {
            aggregateState.status = Failed;
        } else if (processingCount > 0) {
            aggregateState.status = Processing;
            aggregateState.progress = totalProgress / blockCount;
        } else {
            aggregateState.status = Pending;
        }
        
        // Draw block
        QColor blockColor = getBlockColor(aggregateState);
        painter.fillRect(blockRect, blockColor);
        
        // Draw border
        painter.setPen(Qt::black);
        painter.drawRect(blockRect);
        
        // For processing blocks, show progress fill
        if (aggregateState.status == Processing && aggregateState.progress > 0.0f) {
            int fillHeight = static_cast<int>(blockHeight * aggregateState.progress);
            QRect fillRect(x, y + blockHeight - fillHeight, blockWidth, fillHeight);
            painter.fillRect(fillRect, QColor(100, 200, 100, 180));
        }
    }
}

QColor ProgressWidget::getBlockColor(const BlockState &block) const
{
    switch (block.status) {
        case Pending:
            return QColor(200, 200, 200);  // Gray
        case Processing:
            return QColor(255, 200, 100);  // Orange
        case Complete:
            return QColor(100, 200, 100);  // Green
        case Failed:
            return QColor(200, 100, 100);  // Red
        default:
            return QColor(200, 200, 200);
    }
}

void ProgressWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    calculateBlockGrid();
    m_needsRepaint = true;
}

void ProgressWidget::mouseMoveEvent(QMouseEvent *event)
{
    QWidget::mouseMoveEvent(event);
    
    // Show tooltip with block info if hovering over a block
    int blockIndex = getBlockAtPosition(event->pos());
    if (blockIndex >= 0 && blockIndex < m_blocks.size()) {
        const BlockState &block = m_blocks[blockIndex];
        QString tooltip = QString("Block #%1\n").arg(blockIndex);
        
        switch (block.status) {
            case Pending:
                tooltip += "Status: Pending";
                break;
            case Processing:
                tooltip += QString("Status: Processing (%1%)").arg(static_cast<int>(block.progress * 100));
                break;
            case Complete:
                tooltip += QString("Status: Complete\nRatio: %1%").arg(block.compressionRatio * 100, 0, 'f', 2);
                break;
            case Failed:
                tooltip += "Status: Failed";
                break;
        }
        
        QToolTip::showText(event->globalPos(), tooltip, this);
    }
}

int ProgressWidget::getBlockAtPosition(const QPoint &pos) const
{
    if (!m_blockGridWidget || m_totalBlocks == 0) {
        return -1;
    }
    
    // Convert to block grid widget coordinates
    QPoint gridPos = m_blockGridWidget->mapFromParent(pos);
    QRect gridRect = m_blockGridWidget->rect();
    
    if (!gridRect.contains(gridPos)) {
        return -1;
    }
    
    int padding = 4;
    int blockWidth = (gridRect.width() - padding * (m_blockGridCols + 1)) / m_blockGridCols;
    int blockHeight = (gridRect.height() - padding * (m_blockGridRows + 1)) / m_blockGridRows;
    
    blockWidth = qMax(blockWidth, 10);
    blockHeight = qMax(blockHeight, 10);
    
    // Calculate which block was clicked
    int col = (gridPos.x() - padding) / (blockWidth + padding);
    int row = (gridPos.y() - padding) / (blockHeight + padding);
    
    if (col < 0 || col >= m_blockGridCols || row < 0 || row >= m_blockGridRows) {
        return -1;
    }
    
    int displayBlockIndex = row * m_blockGridCols + col;
    int actualBlockIndex = displayBlockIndex * m_blocksPerAggregate;
    
    if (actualBlockIndex >= m_totalBlocks) {
        return -1;
    }
    
    return actualBlockIndex;
}

void ProgressWidget::onUpdateTimer()
{
    if (m_needsRepaint && m_blockGridWidget) {
        m_blockGridWidget->update();
        m_needsRepaint = false;
    }
}

QString ProgressWidget::formatBytes(uint64_t bytes) const
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

QString ProgressWidget::formatTime(int seconds) const
{
    if (seconds < 0) {
        return "--";
    }
    
    if (seconds < 60) {
        return QString("%1s").arg(seconds);
    } else if (seconds < 3600) {
        int mins = seconds / 60;
        int secs = seconds % 60;
        return QString("%1m %2s").arg(mins).arg(secs);
    } else {
        int hours = seconds / 3600;
        int mins = (seconds % 3600) / 60;
        return QString("%1h %2m").arg(hours).arg(mins);
    }
}

