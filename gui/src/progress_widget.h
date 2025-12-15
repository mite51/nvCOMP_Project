/**
 * @file progress_widget.h
 * @brief Advanced progress widget with block-level visualization
 * 
 * Provides granular, block-level progress feedback with visual block grid,
 * real-time throughput, and ETA calculation.
 */

#ifndef PROGRESS_WIDGET_H
#define PROGRESS_WIDGET_H

#include <QWidget>
#include <QTimer>
#include <QLabel>
#include <QProgressBar>
#include <QVector>
#include <QPainter>
#include <QString>

/**
 * @class ProgressWidget
 * @brief Advanced progress visualization with block-level detail
 * 
 * Displays:
 * - Individual block/chunk compression status
 * - Per-block compression ratios
 * - Real-time throughput and ETA
 * - Visual grid of block completion
 */
class ProgressWidget : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief Block status enumeration
     */
    enum BlockStatus {
        Pending,     ///< Not yet processed
        Processing,  ///< Currently being processed
        Complete,    ///< Successfully completed
        Failed       ///< Processing failed
    };

    /**
     * @brief Constructs the progress widget
     * @param parent Parent widget
     */
    explicit ProgressWidget(QWidget *parent = nullptr);
    
    /**
     * @brief Destroys the progress widget
     */
    ~ProgressWidget();

public slots:
    /**
     * @brief Sets the total number of blocks
     * @param total Total block count
     */
    void setTotalBlocks(int total);
    
    /**
     * @brief Updates progress for a specific block
     * @param blockIndex Block index (0-based)
     * @param progress Progress for this block (0.0 to 1.0)
     */
    void updateBlockProgress(int blockIndex, float progress);
    
    /**
     * @brief Marks a block as complete
     * @param blockIndex Block index (0-based)
     * @param compressionRatio Compression ratio for this block
     */
    void setBlockComplete(int blockIndex, float compressionRatio);
    
    /**
     * @brief Marks a block as failed
     * @param blockIndex Block index (0-based)
     */
    void setBlockFailed(int blockIndex);
    
    /**
     * @brief Updates overall progress
     * @param progress Overall progress (0.0 to 1.0)
     */
    void updateOverallProgress(float progress);
    
    /**
     * @brief Updates throughput information
     * @param mbps Throughput in MB/s
     */
    void updateThroughput(double mbps);
    
    /**
     * @brief Sets the current processing stage
     * @param stage Stage name (e.g., "Preparing", "Compressing", "Writing")
     */
    void setCurrentStage(const QString &stage);
    
    /**
     * @brief Sets the current file being processed
     * @param filename File name
     */
    void setCurrentFile(const QString &filename);
    
    /**
     * @brief Updates data processed/total
     * @param current Current bytes processed
     * @param total Total bytes to process
     */
    void setDataProgress(uint64_t current, uint64_t total);
    
    /**
     * @brief Updates estimated time remaining
     * @param seconds Seconds remaining
     */
    void setETA(int seconds);
    
    /**
     * @brief Resets the widget to initial state
     */
    void reset();
    
    /**
     * @brief Draws the block grid (public for custom widget)
     * @param painter QPainter for drawing
     * @param rect Rectangle to draw in
     */
    void drawBlockGrid(QPainter &painter, const QRect &rect);

protected:
    /**
     * @brief Custom paint event for block grid
     * @param event Paint event
     */
    void paintEvent(QPaintEvent *event) override;
    
    /**
     * @brief Handles resize events
     * @param event Resize event
     */
    void resizeEvent(QResizeEvent *event) override;
    
    /**
     * @brief Handles mouse move for tooltips
     * @param event Mouse event
     */
    void mouseMoveEvent(QMouseEvent *event) override;

private:
    /**
     * @brief Structure representing a single block's state
     */
    struct BlockState {
        BlockStatus status;
        float progress;           // 0.0 to 1.0
        float compressionRatio;   // Compressed/Uncompressed
        
        BlockState() : status(Pending), progress(0.0f), compressionRatio(0.0f) {}
    };
    
    // Block data
    QVector<BlockState> m_blocks;
    int m_totalBlocks;
    int m_completedBlocks;
    int m_currentBlock;
    
    // Progress data
    float m_overallProgress;
    double m_throughput;
    QString m_currentStage;
    QString m_currentFile;
    uint64_t m_currentBytes;
    uint64_t m_totalBytes;
    int m_etaSeconds;
    
    // UI components
    QLabel *m_titleLabel;
    QProgressBar *m_overallProgressBar;
    QLabel *m_stageLabel;
    QLabel *m_fileLabel;
    QLabel *m_dataLabel;
    QLabel *m_speedLabel;
    QLabel *m_etaLabel;
    QLabel *m_blockInfoLabel;
    QWidget *m_blockGridWidget;
    
    // Update throttling
    QTimer *m_updateTimer;
    bool m_needsRepaint;
    
    // Block grid rendering
    int m_blockGridRows;
    int m_blockGridCols;
    int m_maxDisplayBlocks;
    int m_blocksPerAggregate;
    
    /**
     * @brief Initializes UI components
     */
    void setupUI();
    
    /**
     * @brief Calculates optimal block grid layout
     */
    void calculateBlockGrid();
    
    /**
     * @brief Gets color for a block based on status
     * @param block Block state
     * @return QColor for rendering
     */
    QColor getBlockColor(const BlockState &block) const;
    
    /**
     * @brief Formats bytes to human-readable string
     * @param bytes Number of bytes
     * @return Formatted string
     */
    QString formatBytes(uint64_t bytes) const;
    
    /**
     * @brief Formats time duration
     * @param seconds Number of seconds
     * @return Formatted string (e.g., "1m 30s")
     */
    QString formatTime(int seconds) const;
    
    /**
     * @brief Gets block index at screen position
     * @param pos Screen position
     * @return Block index, or -1 if none
     */
    int getBlockAtPosition(const QPoint &pos) const;

private slots:
    /**
     * @brief Timer callback for throttled repaints
     */
    void onUpdateTimer();
};

#endif // PROGRESS_WIDGET_H

