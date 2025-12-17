/**
 * @file main.cpp
 * @brief nvCOMP GUI Application Entry Point
 * 
 * Qt-based graphical interface for NVIDIA nvCOMP compression tool.
 * Provides drag-and-drop compression/decompression with GPU acceleration.
 * 
 * Command-line arguments:
 *   --add-file <path>          Add file/folder to GUI and open (for context menu)
 *   --compress                 Enable compress mode (requires --algorithm)
 *   --algorithm <algo>         Set algorithm: lz4, snappy, zstd, gdeflate, ans, bitcomp
 *   --help                     Show help message
 */

#include <QApplication>
#include <QStyleFactory>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QMessageBox>
#include <QStringList>
#include <QFileInfo>
#include <QTimer>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    // Create Qt application instance
    QApplication app(argc, argv);
    
    // Set application metadata
    QApplication::setApplicationName("nvCOMP GUI");
    QApplication::setApplicationVersion("1.0.0");
    QApplication::setOrganizationName("nvCOMP");
    QApplication::setOrganizationDomain("nvidia.com");
    
    // Use native style on Windows/Linux for better OS integration
    // Falls back to Fusion style if native not available
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    
    // ========================================================================
    // Parse command-line arguments
    // ========================================================================
    
    QCommandLineParser parser;
    parser.setApplicationDescription("nvCOMP - GPU Accelerated Compression Tool");
    parser.addHelpOption();
    parser.addVersionOption();
    
    // Define command-line options
    QCommandLineOption addFileOption(QStringList() << "add-file" << "a",
        "Add file or folder to GUI", "path");
    parser.addOption(addFileOption);
    
    QCommandLineOption compressOption(QStringList() << "compress" << "c",
        "Enable compress mode (for context menu)");
    parser.addOption(compressOption);
    
    QCommandLineOption algorithmOption(QStringList() << "algorithm" << "alg",
        "Compression algorithm: lz4, snappy, zstd, gdeflate, ans, bitcomp", "algorithm");
    parser.addOption(algorithmOption);
    
    // Process arguments
    parser.process(app);
    
    // ========================================================================
    // Extract argument values
    // ========================================================================
    
    QStringList filesToAdd;
    QString algorithm;
    bool autoCompress = false;
    
    // Get files to add (can be specified multiple times or as positional args)
    if (parser.isSet(addFileOption)) {
        QStringList addFilePaths = parser.values(addFileOption);
        for (const QString &path : addFilePaths) {
            QFileInfo info(path);
            if (info.exists()) {
                filesToAdd.append(info.absoluteFilePath());
            }
        }
    }
    
    // Also accept positional arguments as files (for drag-and-drop or command line)
    QStringList positionalArgs = parser.positionalArguments();
    for (const QString &path : positionalArgs) {
        QFileInfo info(path);
        if (info.exists()) {
            filesToAdd.append(info.absoluteFilePath());
        }
    }
    
    // Get algorithm
    if (parser.isSet(algorithmOption)) {
        algorithm = parser.value(algorithmOption).toLower();
        
        // Validate algorithm
        QStringList validAlgorithms = {"lz4", "snappy", "zstd", "gdeflate", "ans", "bitcomp"};
        if (!validAlgorithms.contains(algorithm)) {
            QMessageBox::warning(nullptr, "Invalid Algorithm",
                QString("Unknown algorithm: %1\n\nValid algorithms: %2")
                    .arg(algorithm)
                    .arg(validAlgorithms.join(", ")));
            algorithm.clear();
        }
    }
    
    // Check if auto-compress is requested
    if (parser.isSet(compressOption)) {
        autoCompress = true;
        
        // Compress mode requires files and algorithm
        if (filesToAdd.isEmpty()) {
            QMessageBox::warning(nullptr, "No Files",
                "Compress mode requires at least one file.\n\nUsage: nvcomp-gui --compress --algorithm lz4 <file>");
            autoCompress = false;
        }
        if (algorithm.isEmpty()) {
            QMessageBox::warning(nullptr, "No Algorithm",
                "Compress mode requires --algorithm option.\n\nUsage: nvcomp-gui --compress --algorithm lz4 <file>");
            autoCompress = false;
        }
    }
    
    // ========================================================================
    // Create and show main window
    // ========================================================================
    
    MainWindow window;
    
    // Pre-load files if specified
    if (!filesToAdd.isEmpty()) {
        window.addFilesFromCommandLine(filesToAdd);
    }
    
    // Set algorithm if specified
    if (!algorithm.isEmpty()) {
        window.setAlgorithmFromCommandLine(algorithm);
    }
    
    // Show the window
    window.show();
    
    // Auto-start compression if requested (delayed to allow window to fully initialize)
    if (autoCompress) {
        // Use QTimer::singleShot to delay compression start until event loop is running
        QTimer::singleShot(100, &window, &MainWindow::startCompressionFromCommandLine);
    }
    
    // Enter Qt event loop
    return app.exec();
}

