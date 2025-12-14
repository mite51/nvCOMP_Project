/**
 * @file compression_worker.h
 * @brief Worker thread for compression/decompression operations
 * 
 * Provides a QThread-based worker that performs compression operations
 * in the background without freezing the UI. Supports progress reporting,
 * cancellation, and error handling.
 */

#ifndef COMPRESSION_WORKER_H
#define COMPRESSION_WORKER_H

#include <QThread>
#include <QString>
#include <QStringList>
#include <QAtomicInt>
#include <QMutex>
#include <chrono>
#include "nvcomp_c_api.h"

/**
 * @class CompressionWorker
 * @brief Background worker thread for compression operations
 * 
 * Handles compression and decompression operations in a separate thread
 * to keep the UI responsive. Emits signals for progress updates and
 * completion status.
 */
class CompressionWorker : public QThread
{
    Q_OBJECT

public:
    /**
     * @brief Operation types
     */
    enum OperationType {
        COMPRESS,
        DECOMPRESS
    };

    /**
     * @brief Constructs a compression worker
     * @param parent Parent QObject
     */
    explicit CompressionWorker(QObject *parent = nullptr);
    
    /**
     * @brief Destroys the worker and cleans up resources
     */
    ~CompressionWorker();

    /**
     * @brief Configures compression operation
     * @param files List of input files to compress
     * @param outputPath Output archive path
     * @param algorithm Algorithm to use (e.g., "lz4", "snappy", "zstd")
     * @param useCpuMode Use CPU instead of GPU
     * @param volumeSize Maximum volume size (0 for no splitting)
     */
    void setupCompress(const QStringList &files,
                      const QString &outputPath,
                      const QString &algorithm,
                      bool useCpuMode,
                      uint64_t volumeSize = 0);

    /**
     * @brief Configures decompression operation
     * @param files List of compressed files to decompress
     * @param outputPath Output directory path
     * @param algorithm Algorithm to use (auto-detect if empty)
     * @param useCpuMode Use CPU instead of GPU
     */
    void setupDecompress(const QStringList &files,
                        const QString &outputPath,
                        const QString &algorithm,
                        bool useCpuMode);

    /**
     * @brief Requests cancellation of the current operation
     * 
     * The operation will stop at the next safe checkpoint.
     * The canceled() signal will be emitted.
     */
    void cancel();

    /**
     * @brief Checks if the operation was canceled
     * @return true if cancellation was requested
     */
    bool isCanceled() const;

signals:
    /**
     * @brief Emitted when progress is updated
     * @param percentage Progress percentage (0-100)
     * @param currentFile Currently processing file
     */
    void progressChanged(int percentage, const QString &currentFile);

    /**
     * @brief Emitted with detailed progress information
     * @param current Current bytes processed
     * @param total Total bytes to process
     * @param speedMBps Processing speed in MB/s
     * @param etaSeconds Estimated time remaining in seconds
     */
    void progressDetails(uint64_t current, uint64_t total, double speedMBps, int etaSeconds);

    /**
     * @brief Emitted when operation completes successfully
     * @param outputPath Path to the output file or directory
     * @param compressionRatio Compression ratio (compressed/uncompressed), 0 for decompress
     * @param durationMs Operation duration in milliseconds
     */
    void finished(const QString &outputPath, double compressionRatio, qint64 durationMs);

    /**
     * @brief Emitted when operation fails
     * @param errorMessage Error description
     */
    void error(const QString &errorMessage);

    /**
     * @brief Emitted when operation is canceled
     */
    void canceled();

    /**
     * @brief Emitted with status messages
     * @param message Status message for display
     */
    void statusMessage(const QString &message);

protected:
    /**
     * @brief Main thread execution function
     * 
     * Performs the configured operation in the background.
     * Called automatically by QThread::start().
     */
    void run() override;

private:
    // Operation configuration
    OperationType m_operationType;
    QStringList m_inputFiles;
    QString m_outputPath;
    QString m_algorithm;
    bool m_useCpuMode;
    uint64_t m_volumeSize;

    // Progress tracking
    QAtomicInt m_canceled;
    uint64_t m_totalBytes;
    uint64_t m_processedBytes;
    std::chrono::steady_clock::time_point m_startTime;
    mutable QMutex m_mutex;

    // nvCOMP operation handle
    nvcomp_operation_handle m_operationHandle;

    /**
     * @brief Performs compression operation
     */
    void performCompress();

    /**
     * @brief Performs decompression operation
     */
    void performDecompress();

    /**
     * @brief Progress callback for nvCOMP C API
     * @param current Current progress value
     * @param total Total progress value
     * @param user_data Pointer to CompressionWorker instance
     */
    static void progressCallback(uint64_t current, uint64_t total, void* user_data);

    /**
     * @brief Updates progress information
     * @param current Current bytes processed
     * @param total Total bytes to process
     */
    void updateProgress(uint64_t current, uint64_t total);

    /**
     * @brief Calculates processing speed
     * @return Speed in MB/s
     */
    double calculateSpeed() const;

    /**
     * @brief Estimates time remaining
     * @return Estimated seconds remaining
     */
    int estimateTimeRemaining() const;

    /**
     * @brief Converts Qt algorithm string to nvCOMP algorithm enum
     * @param algorithm Algorithm string
     * @return nvCOMP algorithm type
     */
    nvcomp_algorithm_t algorithmStringToEnum(const QString &algorithm) const;

    /**
     * @brief Gets total size of all input files
     * @return Total size in bytes
     */
    uint64_t calculateTotalSize() const;
};

#endif // COMPRESSION_WORKER_H

