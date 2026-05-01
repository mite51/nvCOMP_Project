/**
 * @file compression_worker.cpp
 * @brief Implementation of CompressionWorker class
 */

#include "compression_worker.h"
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>
#include <QMetaType>
#include <QDebug>
#include <cstring>
#include <vector>
#include <string>

CompressionWorker::CompressionWorker(QObject *parent)
    : QThread(parent)
    , m_operationType(COMPRESS)
    , m_useCpuMode(false)
    , m_volumeSize(0)
    , m_canceled(0)
    , m_finalElapsedMs(0)
    , m_lastEmittedPercent(-1)
    , m_lastEmittedStage()
{
    // Register the stats struct as a Qt metatype so it can cross threads via
    // queued signals. Calling qRegisterMetaType is idempotent.
    qRegisterMetaType<nvcomp_compression_stats_t>("nvcomp_compression_stats_t");
}

CompressionWorker::~CompressionWorker()
{
    if (isRunning()) {
        cancel();
        if (!wait(5000)) {
            terminate();
            wait();
        }
    }
}

void CompressionWorker::setupCompress(const QStringList &paths,
                                      const QString &outputPath,
                                      const QString &algorithm,
                                      bool useCpuMode,
                                      uint64_t volumeSize)
{
    QMutexLocker locker(&m_mutex);
    m_operationType = COMPRESS;
    m_inputPaths = paths;
    m_outputPath = outputPath;
    m_algorithm = algorithm;
    m_useCpuMode = useCpuMode;
    m_volumeSize = volumeSize;
    m_canceled = 0;
    m_finalElapsedMs = 0;
}

void CompressionWorker::setupDecompress(const QStringList &files,
                                        const QString &outputPath,
                                        const QString &algorithm,
                                        bool useCpuMode)
{
    QMutexLocker locker(&m_mutex);
    m_operationType = DECOMPRESS;
    m_inputPaths = files;
    m_outputPath = outputPath;
    m_algorithm = algorithm;
    m_useCpuMode = useCpuMode;
    m_volumeSize = 0;
    m_canceled = 0;
    m_finalElapsedMs = 0;
}

void CompressionWorker::cancel()
{
    m_canceled = 1;
}

bool CompressionWorker::isCanceled() const
{
    return m_canceled.loadAcquire() != 0;
}

qint64 CompressionWorker::getElapsedTime() const
{
    if (isRunning()) {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - m_startTime).count();
    }
    return m_finalElapsedMs;
}

void CompressionWorker::run()
{
    m_startTime = std::chrono::steady_clock::now();
    // Reset throttle state for this run (do NOT use process statics).
    m_lastEmittedPercent = -1;
    m_lastEmittedStage.clear();
    m_lastEmitTime = std::chrono::steady_clock::time_point::min();

    try {
        if (m_operationType == COMPRESS) {
            performCompress();
        } else {
            performDecompress();
        }
    } catch (const std::exception &e) {
        emit error(QString("Exception: %1").arg(e.what()));
    } catch (...) {
        emit error("Unknown error occurred");
    }

    auto endTime = std::chrono::steady_clock::now();
    m_finalElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                           endTime - m_startTime).count();
}

void CompressionWorker::performCompress()
{
    if (m_inputPaths.isEmpty()) {
        emit error("No files selected for compression");
        return;
    }

    nvcomp_algorithm_t algo = algorithmStringToEnum(m_algorithm);
    if (algo == NVCOMP_ALGO_UNKNOWN) {
        emit error(QString("Unknown algorithm: %1").arg(m_algorithm));
        return;
    }

    // Resolve output path. Match the CLI's defaulting behavior for parity.
    QString outputFile = m_outputPath;
    if (outputFile.isEmpty()) {
        QFileInfo firstFile(m_inputPaths.first());
        if (m_inputPaths.size() == 1) {
            outputFile = firstFile.absolutePath() + "/" + firstFile.baseName() + ".nvcomp";
        } else {
            outputFile = firstFile.absolutePath() + "/archive.nvcomp";
        }
    }

    emit statusMessage(QString("Output: %1").arg(outputFile));

    if (isCanceled()) {
        emit canceled();
        return;
    }

    if (m_inputPaths.size() == 1) {
        emit statusMessage(QString("Compressing: %1")
                               .arg(QFileInfo(m_inputPaths.first()).fileName()));
    } else {
        emit statusMessage(QString("Compressing %1 items...").arg(m_inputPaths.size()));
    }

    // Set up the operation handle once. The throttling lives in the core
    // (makeThrottledCallback wraps our callback there), so we do NOT need to
    // re-throttle here - we can deliver every call straight to the UI signal.
    nvcomp_operation_handle handle = nvcomp_create_operation_handle();
    if (handle) {
        nvcomp_set_block_progress_callback(handle, &CompressionWorker::blockProgressCallback, this);
    }

    nvcomp_compression_stats_t stats;
    std::memset(&stats, 0, sizeof(stats));

    // Convert input paths to UTF-8 ONCE and keep them in scope for the duration
    // of the call.
    std::string outputFileStr = outputFile.toStdString();

    nvcomp_error_t result = NVCOMP_SUCCESS;

    if (m_inputPaths.size() == 1) {
        // Single path (file OR folder) - use the same single-path API the CLI
        // uses. The core handles directories internally.
        std::string inputPathStr = m_inputPaths.first().toStdString();

        if (m_useCpuMode) {
            result = nvcomp_compress_cpu(handle, algo,
                                          inputPathStr.c_str(), outputFileStr.c_str(),
                                          m_volumeSize, &stats);
        } else if (nvcomp_is_cross_compatible(algo)) {
            result = nvcomp_compress_gpu_batched(handle, algo,
                                                  inputPathStr.c_str(), outputFileStr.c_str(),
                                                  m_volumeSize, &stats);
        } else {
            result = nvcomp_compress_gpu_manager(handle, algo,
                                                  inputPathStr.c_str(), outputFileStr.c_str(),
                                                  m_volumeSize, &stats);
        }
    } else {
        // Multiple paths - use file-list API.
        std::vector<std::string> filePathStrings;
        std::vector<const char *> filePaths;
        filePathStrings.reserve(m_inputPaths.size());
        filePaths.reserve(m_inputPaths.size());

        for (const QString &p : m_inputPaths) {
            filePathStrings.push_back(p.toStdString());
        }
        for (const std::string &s : filePathStrings) {
            filePaths.push_back(s.c_str());
        }

        if (m_useCpuMode) {
            result = nvcomp_compress_cpu_file_list(handle, algo,
                                                    filePaths.data(), filePaths.size(),
                                                    outputFileStr.c_str(),
                                                    m_volumeSize, &stats);
        } else if (nvcomp_is_cross_compatible(algo)) {
            result = nvcomp_compress_gpu_batched_file_list(handle, algo,
                                                            filePaths.data(), filePaths.size(),
                                                            outputFileStr.c_str(),
                                                            m_volumeSize, &stats);
        } else {
            result = nvcomp_compress_gpu_manager_file_list(handle, algo,
                                                            filePaths.data(), filePaths.size(),
                                                            outputFileStr.c_str(),
                                                            m_volumeSize, &stats);
        }
    }

    if (handle) nvcomp_destroy_operation_handle(handle);

    if (result != NVCOMP_SUCCESS) {
        const char *errorMsg = nvcomp_get_last_error();
        emit error(QString("Compression failed: %1")
                       .arg(errorMsg && *errorMsg ? errorMsg : "Unknown error"));
        return;
    }

    if (isCanceled()) {
        emit canceled();
        return;
    }

    // Final 100% progress, then finished.
    qint64 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - m_startTime).count();
    emit progressUpdate(100, QStringLiteral("complete"),
                        stats.throughput_mbps, elapsed);
    emit finished(outputFile, stats);
}

void CompressionWorker::performDecompress()
{
    if (m_inputPaths.isEmpty()) {
        emit error("No files selected for decompression");
        return;
    }

    QString outputPath = m_outputPath;
    if (outputPath.isEmpty()) {
        outputPath = QFileInfo(m_inputPaths.first()).absolutePath();
    }
    emit statusMessage(QString("Output directory: %1").arg(outputPath));

    nvcomp_operation_handle handle = nvcomp_create_operation_handle();
    if (handle) {
        nvcomp_set_block_progress_callback(handle, &CompressionWorker::blockProgressCallback, this);
    }

    // Aggregate stats across all input archives.
    nvcomp_compression_stats_t aggStats;
    std::memset(&aggStats, 0, sizeof(aggStats));

    for (int idx = 0; idx < m_inputPaths.size(); ++idx) {
        if (isCanceled()) {
            if (handle) nvcomp_destroy_operation_handle(handle);
            emit canceled();
            return;
        }

        const QString &inputFile = m_inputPaths.at(idx);
        QFileInfo fileInfo(inputFile);
        emit statusMessage(QString("Decompressing: %1").arg(fileInfo.fileName()));

        nvcomp_algorithm_t algo = NVCOMP_ALGO_UNKNOWN;
        if (!m_algorithm.isEmpty()) {
            algo = algorithmStringToEnum(m_algorithm);
        } else {
            std::string fp = inputFile.toStdString();
            algo = nvcomp_detect_algorithm_from_file(fp.c_str());
        }

        if (algo == NVCOMP_ALGO_UNKNOWN) {
            emit error(QString("Could not detect algorithm for: %1").arg(fileInfo.fileName()));
            continue;
        }

        std::string inputFileStr = inputFile.toStdString();
        std::string outputPathStr = outputPath.toStdString();

        nvcomp_compression_stats_t stats;
        std::memset(&stats, 0, sizeof(stats));

        nvcomp_error_t result;
        if (m_useCpuMode) {
            result = nvcomp_decompress_cpu(handle, algo,
                                            inputFileStr.c_str(), outputPathStr.c_str(),
                                            &stats);
        } else if (nvcomp_is_cross_compatible(algo)) {
            result = nvcomp_decompress_gpu_batched(handle, algo,
                                                    inputFileStr.c_str(), outputPathStr.c_str(),
                                                    &stats);
        } else {
            result = nvcomp_decompress_gpu_manager(handle,
                                                    inputFileStr.c_str(), outputPathStr.c_str(),
                                                    &stats);
        }

        if (result != NVCOMP_SUCCESS) {
            const char *errorMsg = nvcomp_get_last_error();
            emit error(QString("Decompression failed for %1: %2")
                           .arg(fileInfo.fileName())
                           .arg(errorMsg && *errorMsg ? errorMsg : "Unknown error"));
            continue;
        }

        // Accumulate.
        aggStats.read_sec += stats.read_sec;
        aggStats.prepare_sec += stats.prepare_sec;
        aggStats.compute_sec += stats.compute_sec;
        aggStats.write_sec += stats.write_sec;
        aggStats.total_sec += stats.total_sec;
        aggStats.input_bytes += stats.input_bytes;
        aggStats.output_bytes += stats.output_bytes;
    }

    if (handle) nvcomp_destroy_operation_handle(handle);

    // Re-derive throughput/ratio from aggregated bytes & total time.
    if (aggStats.total_sec > 0.0 && aggStats.input_bytes > 0) {
        double mb = static_cast<double>(aggStats.input_bytes) / (1024.0 * 1024.0);
        aggStats.throughput_mbps = mb / aggStats.total_sec;
        aggStats.throughput_gbps = aggStats.throughput_mbps / 1024.0;
    }
    if (aggStats.output_bytes > 0) {
        aggStats.ratio = static_cast<double>(aggStats.input_bytes)
                       / static_cast<double>(aggStats.output_bytes);
    }

    qint64 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - m_startTime).count();
    emit progressUpdate(100, QStringLiteral("complete"),
                        aggStats.throughput_mbps, elapsed);
    emit finished(outputPath, aggStats);
}

void CompressionWorker::blockProgressCallback(nvcomp_operation_handle /*handle*/,
                                              const nvcomp_progress_info_t *info,
                                              void *user_data)
{
    if (!user_data || !info) return;
    auto *worker = static_cast<CompressionWorker *>(user_data);

    if (worker->isCanceled()) return;

    // The core already throttles us to ~30Hz / 1% pct. Apply a second
    // throttle here aimed at the UI: cap to ~10Hz unless the percent or stage
    // changed. This keeps the Qt event queue calm even if the core's throttle
    // ever loosens.
    const auto now = std::chrono::steady_clock::now();
    const int pct = std::max(0, std::min(100,
                       static_cast<int>(info->overallProgress * 100.0f)));
    const QString stage = info->stage ? QString::fromUtf8(info->stage) : QString();

    const bool isTerminal = info->overallProgress >= 1.0f;
    const bool stageChanged = stage != worker->m_lastEmittedStage;
    const bool pctChanged = pct != worker->m_lastEmittedPercent;
    const bool timeOk = (now - worker->m_lastEmitTime) >= std::chrono::milliseconds(100);

    if (!(isTerminal || stageChanged || (pctChanged && timeOk))) return;

    worker->m_lastEmittedPercent = pct;
    worker->m_lastEmittedStage = stage;
    worker->m_lastEmitTime = now;

    qint64 elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - worker->m_startTime).count();

    emit worker->progressUpdate(pct, stage, info->throughputMBps, elapsedMs);
}

nvcomp_algorithm_t CompressionWorker::algorithmStringToEnum(const QString &algorithm) const
{
    QString algoLower = algorithm.toLower();
    if (algoLower == "lz4")      return NVCOMP_ALGO_LZ4;
    if (algoLower == "snappy")   return NVCOMP_ALGO_SNAPPY;
    if (algoLower == "zstd")     return NVCOMP_ALGO_ZSTD;
    if (algoLower == "gdeflate") return NVCOMP_ALGO_GDEFLATE;
    if (algoLower == "ans")      return NVCOMP_ALGO_ANS;
    if (algoLower == "bitcomp")  return NVCOMP_ALGO_BITCOMP;
    return NVCOMP_ALGO_UNKNOWN;
}
