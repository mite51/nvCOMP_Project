/**
 * @file compression_worker.cpp
 * @brief Implementation of CompressionWorker class
 */

#include "compression_worker.h"
#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <algorithm>

CompressionWorker::CompressionWorker(QObject *parent)
    : QThread(parent)
    , m_operationType(COMPRESS)
    , m_useCpuMode(false)
    , m_volumeSize(0)
    , m_canceled(0)
    , m_totalBytes(0)
    , m_processedBytes(0)
    , m_operationHandle(nullptr)
{
}

CompressionWorker::~CompressionWorker()
{
    // Ensure thread is stopped before destruction
    if (isRunning()) {
        cancel();
        // Wait up to 5 seconds for thread to finish
        if (!wait(5000)) {
            // Thread didn't finish gracefully, force terminate
            terminate();
            wait();  // Wait for termination to complete
        }
    }

    // Clean up operation handle if it exists
    if (m_operationHandle) {
        nvcomp_destroy_operation_handle(m_operationHandle);
        m_operationHandle = nullptr;
    }
}

void CompressionWorker::setupCompress(const QStringList &files,
                                     const QString &outputPath,
                                     const QString &algorithm,
                                     bool useCpuMode,
                                     uint64_t volumeSize)
{
    QMutexLocker locker(&m_mutex);
    m_operationType = COMPRESS;
    m_inputFiles = files;
    m_outputPath = outputPath;
    m_algorithm = algorithm;
    m_useCpuMode = useCpuMode;
    m_volumeSize = volumeSize;
    m_canceled = 0;
    m_totalBytes = calculateTotalSize();
    m_processedBytes = 0;
}

void CompressionWorker::setupDecompress(const QStringList &files,
                                       const QString &outputPath,
                                       const QString &algorithm,
                                       bool useCpuMode)
{
    QMutexLocker locker(&m_mutex);
    m_operationType = DECOMPRESS;
    m_inputFiles = files;
    m_outputPath = outputPath;
    m_algorithm = algorithm;
    m_useCpuMode = useCpuMode;
    m_volumeSize = 0;
    m_canceled = 0;
    m_totalBytes = calculateTotalSize();
    m_processedBytes = 0;
}

void CompressionWorker::cancel()
{
    m_canceled = 1;
    qDebug() << "Cancellation requested";
}

bool CompressionWorker::isCanceled() const
{
    return m_canceled.loadAcquire() != 0;
}

void CompressionWorker::run()
{
    m_startTime = std::chrono::steady_clock::now();
    
    // Create operation handle for progress tracking
    m_operationHandle = nvcomp_create_operation_handle();
    if (m_operationHandle) {
        nvcomp_set_progress_callback(m_operationHandle, progressCallback, this);
    }

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

    // Clean up operation handle
    if (m_operationHandle) {
        nvcomp_destroy_operation_handle(m_operationHandle);
        m_operationHandle = nullptr;
    }
}

void CompressionWorker::performCompress()
{
    if (m_inputFiles.isEmpty()) {
        emit error("No files selected for compression");
        return;
    }

    emit statusMessage("Starting compression...");

    nvcomp_algorithm_t algo = algorithmStringToEnum(m_algorithm);
    if (algo == NVCOMP_ALGO_UNKNOWN) {
        emit error(QString("Unknown algorithm: %1").arg(m_algorithm));
        return;
    }

    // Determine output path
    QString outputFile = m_outputPath;
    if (outputFile.isEmpty()) {
        // Auto-generate output filename
        QFileInfo firstFile(m_inputFiles.first());
        if (m_inputFiles.size() == 1) {
            outputFile = firstFile.absolutePath() + "/" + firstFile.baseName() + ".nvcomp";
        } else {
            outputFile = firstFile.absolutePath() + "/archive.nvcomp";
        }
    }

    emit statusMessage(QString("Output: %1").arg(outputFile));

    // Calculate total size
    uint64_t totalUncompressedSize = 0;
    for (const QString &filePath : m_inputFiles) {
        QFileInfo fileInfo(filePath);
        if (fileInfo.isFile()) {
            totalUncompressedSize += fileInfo.size();
        }
    }

    // Compress files (single or multiple)
    if (isCanceled()) {
        emit canceled();
        return;
    }

    if (m_inputFiles.size() == 1) {
        emit statusMessage(QString("Compressing: %1").arg(QFileInfo(m_inputFiles.first()).fileName()));
    } else {
        emit statusMessage(QString("Compressing %1 files...").arg(m_inputFiles.size()));
    }
    
    // Show initial progress
    emit progressChanged(10, QString("Preparing..."));
    QThread::msleep(100);  // Brief delay so user sees the UI update

    nvcomp_error_t result;
    
    // Show progress before compression starts
    emit progressChanged(25, QString("Compressing..."));
    
    // Convert QString to std::string ONCE and keep them in scope
    std::string inputPathStr;
    std::string outputFileStr = outputFile.toStdString();
    
    // Use file list API for multiple files, or single file API for one file
    if (m_inputFiles.size() == 1) {
        inputPathStr = m_inputFiles.first().toStdString();
        
        // Single file - use original single-file API
        if (m_useCpuMode) {
            result = nvcomp_compress_cpu(
                m_operationHandle,
                algo,
                inputPathStr.c_str(),
                outputFileStr.c_str(),
                m_volumeSize
            );
        } else {
            if (nvcomp_is_cross_compatible(algo)) {
                result = nvcomp_compress_gpu_batched(
                    m_operationHandle,
                    algo,
                    inputPathStr.c_str(),
                    outputFileStr.c_str(),
                    m_volumeSize
                );
            } else {
                result = nvcomp_compress_gpu_manager(
                    m_operationHandle,
                    algo,
                    inputPathStr.c_str(),
                    outputFileStr.c_str(),
                    m_volumeSize
                );
            }
        }
    } else {
        // Multiple files - use file list API
        // Convert QStringList to std::vector<std::string> and then to const char**
        std::vector<std::string> filePathStrings;
        std::vector<const char*> filePaths;
        filePathStrings.reserve(m_inputFiles.size());
        filePaths.reserve(m_inputFiles.size());
        
        for (const QString &filePath : m_inputFiles) {
            filePathStrings.push_back(filePath.toStdString());
        }
        
        for (const std::string &pathStr : filePathStrings) {
            filePaths.push_back(pathStr.c_str());
        }
        
        if (m_useCpuMode) {
            result = nvcomp_compress_cpu_file_list(
                m_operationHandle,
                algo,
                filePaths.data(),
                filePaths.size(),
                outputFileStr.c_str(),
                m_volumeSize
            );
        } else {
            if (nvcomp_is_cross_compatible(algo)) {
                result = nvcomp_compress_gpu_batched_file_list(
                    m_operationHandle,
                    algo,
                    filePaths.data(),
                    filePaths.size(),
                    outputFileStr.c_str(),
                    m_volumeSize
                );
            } else {
                result = nvcomp_compress_gpu_manager_file_list(
                    m_operationHandle,
                    algo,
                    filePaths.data(),
                    filePaths.size(),
                    outputFileStr.c_str(),
                    m_volumeSize
                );
            }
        }
    }

    // Show progress after compression (core functions are synchronous/blocking)
    emit progressChanged(90, QString("Finalizing..."));
    
    if (result != NVCOMP_SUCCESS) {
        const char* errorMsg = nvcomp_get_last_error();
        emit error(QString("Compression failed: %1").arg(errorMsg ? errorMsg : "Unknown error"));
        return;
    }

    if (isCanceled()) {
        emit canceled();
        return;
    }

    // Calculate compression ratio
    QFileInfo outputInfo(outputFile);
    double compressionRatio = 0.0;
    if (totalUncompressedSize > 0 && outputInfo.exists()) {
        compressionRatio = static_cast<double>(outputInfo.size()) / static_cast<double>(totalUncompressedSize);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_startTime);

    emit progressChanged(100, "Complete");
    emit finished(outputFile, compressionRatio, duration.count());
}

void CompressionWorker::performDecompress()
{
    if (m_inputFiles.isEmpty()) {
        emit error("No files selected for decompression");
        return;
    }

    emit statusMessage("Starting decompression...");

    // Determine output path
    QString outputPath = m_outputPath;
    if (outputPath.isEmpty()) {
        // Use same directory as input file
        QFileInfo inputInfo(m_inputFiles.first());
        outputPath = inputInfo.absolutePath();
    }

    emit statusMessage(QString("Output directory: %1").arg(outputPath));

    // Process each file
    for (const QString &inputFile : m_inputFiles) {
        if (isCanceled()) {
            emit canceled();
            return;
        }

        QFileInfo fileInfo(inputFile);
        emit statusMessage(QString("Decompressing: %1").arg(fileInfo.fileName()));
        emit progressChanged(0, fileInfo.fileName());

        // Detect algorithm if not specified
        nvcomp_algorithm_t algo = NVCOMP_ALGO_UNKNOWN;
        if (!m_algorithm.isEmpty()) {
            algo = algorithmStringToEnum(m_algorithm);
        } else {
            algo = nvcomp_detect_algorithm_from_file(inputFile.toStdString().c_str());
        }

        if (algo == NVCOMP_ALGO_UNKNOWN) {
            emit error(QString("Could not detect algorithm for: %1").arg(fileInfo.fileName()));
            continue;
        }

        // Construct output path
        QString outputFilePath = outputPath + "/" + fileInfo.completeBaseName();

        nvcomp_error_t result;
        if (m_useCpuMode) {
            result = nvcomp_decompress_cpu(
                m_operationHandle,
                algo,
                inputFile.toStdString().c_str(),
                outputFilePath.toStdString().c_str()
            );
        } else {
            if (nvcomp_is_cross_compatible(algo)) {
                result = nvcomp_decompress_gpu_batched(
                    m_operationHandle,
                    algo,
                    inputFile.toStdString().c_str(),
                    outputFilePath.toStdString().c_str()
                );
            } else {
                result = nvcomp_decompress_gpu_manager(
                    m_operationHandle,
                    inputFile.toStdString().c_str(),
                    outputFilePath.toStdString().c_str()
                );
            }
        }

        if (result != NVCOMP_SUCCESS) {
            const char* errorMsg = nvcomp_get_last_error();
            emit error(QString("Decompression failed for %1: %2")
                      .arg(fileInfo.fileName())
                      .arg(errorMsg ? errorMsg : "Unknown error"));
            continue;
        }

        if (isCanceled()) {
            emit canceled();
            return;
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_startTime);

    emit progressChanged(100, "Complete");
    emit finished(outputPath, 0.0, duration.count());
}

void CompressionWorker::progressCallback(uint64_t current, uint64_t total, void* user_data)
{
    if (user_data) {
        CompressionWorker* worker = static_cast<CompressionWorker*>(user_data);
        worker->updateProgress(current, total);
    }
}

void CompressionWorker::updateProgress(uint64_t current, uint64_t total)
{
    if (isCanceled()) {
        return;
    }

    m_processedBytes = current;
    m_totalBytes = total;

    // Calculate percentage
    int percentage = 0;
    if (total > 0) {
        percentage = static_cast<int>((current * 100) / total);
    }

    // Calculate speed and ETA
    double speedMBps = calculateSpeed();
    int etaSeconds = estimateTimeRemaining();

    // Emit progress signals
    emit progressChanged(percentage, QString());
    emit progressDetails(current, total, speedMBps, etaSeconds);
}

double CompressionWorker::calculateSpeed() const
{
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - m_startTime);
    
    if (elapsed.count() == 0) {
        return 0.0;
    }

    // Speed in MB/s
    double seconds = elapsed.count() / 1000.0;
    double megabytes = m_processedBytes / (1024.0 * 1024.0);
    return megabytes / seconds;
}

int CompressionWorker::estimateTimeRemaining() const
{
    if (m_processedBytes == 0 || m_totalBytes == 0) {
        return 0;
    }

    auto currentTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - m_startTime);
    
    if (elapsed.count() == 0) {
        return 0;
    }

    // Estimate based on current progress
    uint64_t remaining = m_totalBytes - m_processedBytes;
    double ratio = static_cast<double>(remaining) / static_cast<double>(m_processedBytes);
    int etaSeconds = static_cast<int>(elapsed.count() * ratio);

    return etaSeconds;
}

nvcomp_algorithm_t CompressionWorker::algorithmStringToEnum(const QString &algorithm) const
{
    QString algoLower = algorithm.toLower();
    
    if (algoLower == "lz4") return NVCOMP_ALGO_LZ4;
    if (algoLower == "snappy") return NVCOMP_ALGO_SNAPPY;
    if (algoLower == "zstd") return NVCOMP_ALGO_ZSTD;
    if (algoLower == "gdeflate") return NVCOMP_ALGO_GDEFLATE;
    if (algoLower == "ans") return NVCOMP_ALGO_ANS;
    if (algoLower == "bitcomp") return NVCOMP_ALGO_BITCOMP;
    
    return NVCOMP_ALGO_UNKNOWN;
}

uint64_t CompressionWorker::calculateTotalSize() const
{
    uint64_t totalSize = 0;
    
    for (const QString &filePath : m_inputFiles) {
        QFileInfo fileInfo(filePath);
        if (fileInfo.isFile()) {
            totalSize += fileInfo.size();
        } else if (fileInfo.isDir()) {
            // Recursively calculate directory size
            QDir dir(filePath);
            QFileInfoList fileList = dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot, QDir::NoSort);
            for (const QFileInfo &file : fileList) {
                totalSize += file.size();
            }
        }
    }
    
    return totalSize;
}

