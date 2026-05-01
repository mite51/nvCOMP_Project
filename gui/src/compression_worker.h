/**
 * @file compression_worker.h
 * @brief Worker thread for compression/decompression operations
 *
 * A minimal QThread-based worker that:
 *  - Calls exactly one core C API entry point per operation, mirroring the CLI.
 *  - Reports progress via a single throttled signal so the main thread is never
 *    flooded with cross-thread events.
 *  - Returns the per-phase timing struct produced by the core in the finished()
 *    signal so the GUI can render the same numbers the CLI prints.
 *
 * History: a previous version of this worker emitted 5-7 cross-thread Qt
 * signals per GPU chunk (~16K per GB) AND piped them through main-thread slots
 * that called qDebug() and rerendered the status bar per chunk. The result was
 * GUI compression running ~2x slower than the CLI on the same input. The fix
 * lives both here (single throttled signal, no static state) and in the core
 * (deletion of the post-completion per-chunk callback loop and the new
 * makeThrottledCallback helper).
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
 */
class CompressionWorker : public QThread
{
    Q_OBJECT

public:
    enum OperationType {
        COMPRESS,
        DECOMPRESS
    };

    explicit CompressionWorker(QObject *parent = nullptr);
    ~CompressionWorker();

    /**
     * Configures a compression operation. The set of paths matches what the
     * user selected in the file list - both individual files and folders are
     * accepted. The core library will recurse into folders.
     */
    void setupCompress(const QStringList &paths,
                       const QString &outputPath,
                       const QString &algorithm,
                       bool useCpuMode,
                       uint64_t volumeSize = 0);

    /**
     * Configures a decompression operation. Each path is decompressed into
     * outputPath in turn (mirroring how the CLI handles a list of archives).
     */
    void setupDecompress(const QStringList &files,
                         const QString &outputPath,
                         const QString &algorithm,
                         bool useCpuMode);

    /**
     * Requests cancellation. Checked at safe points; the operation may run to
     * completion of the current core call before stopping.
     */
    void cancel();
    bool isCanceled() const;

    /**
     * Elapsed time in milliseconds. While running this is the live elapsed
     * time; after completion it is the final duration recorded by run().
     */
    qint64 getElapsedTime() const;

signals:
    /**
     * Single throttled progress signal. Emitted at most ~10 times per second
     * by run()'s throttled callback, with all the data the UI needs to update.
     */
    void progressUpdate(int percent,
                        const QString &stage,
                        double mbps,
                        qint64 elapsedMs);

    /**
     * Status text suitable for the status bar (start, output path, etc).
     */
    void statusMessage(const QString &message);

    /**
     * Operation succeeded. stats holds the per-phase timing populated by the
     * core; for decompression compressionRatio is 0.
     */
    void finished(const QString &outputPath,
                  const nvcomp_compression_stats_t &stats);

    void error(const QString &errorMessage);
    void canceled();

protected:
    void run() override;

private:
    // Operation configuration (set under m_mutex by setup*())
    OperationType m_operationType;
    QStringList m_inputPaths;
    QString m_outputPath;
    QString m_algorithm;
    bool m_useCpuMode;
    uint64_t m_volumeSize;

    // Live state
    QAtomicInt m_canceled;
    std::chrono::steady_clock::time_point m_startTime;
    qint64 m_finalElapsedMs;
    mutable QMutex m_mutex;

    // Throttle bookkeeping for progressCallback. These live on the worker
    // *instance* (NOT process-statics like the previous implementation) so
    // multiple sequential operations don't poison each other.
    int m_lastEmittedPercent;
    std::chrono::steady_clock::time_point m_lastEmitTime;
    QString m_lastEmittedStage;

    // Implementation
    void performCompress();
    void performDecompress();
    nvcomp_algorithm_t algorithmStringToEnum(const QString &algorithm) const;

    // Single static block-progress callback delivered to the C API. It
    // forwards (with throttling) to the instance's progressUpdate() signal.
    static void blockProgressCallback(nvcomp_operation_handle handle,
                                      const nvcomp_progress_info_t *info,
                                      void *user_data);
};

// Make the C stats struct a Qt-known metatype so it can travel through
// queued signal/slot connections.
Q_DECLARE_METATYPE(nvcomp_compression_stats_t)

#endif // COMPRESSION_WORKER_H
