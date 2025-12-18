/**
 * @file main.cpp
 * @brief nvCOMP GUI Application Entry Point
 * 
 * Qt-based graphical interface for NVIDIA nvCOMP compression tool.
 * Provides drag-and-drop compression/decompression with GPU acceleration.
 * 
 * Command-line arguments:
 *   --add-file <path>                Add file/folder to GUI and open (for context menu)
 *   --compress                       Enable compress mode (requires --algorithm)
 *   --algorithm <algo>               Set algorithm: lz4, snappy, zstd, gdeflate, ans, bitcomp
 *   --register-context-menu          Register Windows Explorer context menu (admin required)
 *   --unregister-context-menu        Unregister Windows Explorer context menu (admin required)
 *   --register-file-associations     Register file associations (admin required)
 *   --unregister-file-associations   Unregister file associations (admin required)
 *   --help                           Show help message
 */

#include <QApplication>
#include <QStyleFactory>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QMessageBox>
#include <QStringList>
#include <QFileInfo>
#include <QTimer>
#include <QCoreApplication>
#include <iostream>
#include "mainwindow.h"

#ifdef _WIN32
#include "../../platform/windows/context_menu.h"
#include "../../platform/windows/file_associations.h"
#endif

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
    
    QCommandLineOption registerContextMenuOption("register-context-menu",
        "Register Windows Explorer context menu (requires administrator privileges)");
    parser.addOption(registerContextMenuOption);
    
    QCommandLineOption unregisterContextMenuOption("unregister-context-menu",
        "Unregister Windows Explorer context menu (requires administrator privileges)");
    parser.addOption(unregisterContextMenuOption);
    
    QCommandLineOption registerFileAssocOption("register-file-associations",
        "Register file associations for nvCOMP archive types (requires administrator privileges)");
    parser.addOption(registerFileAssocOption);
    
    QCommandLineOption unregisterFileAssocOption("unregister-file-associations",
        "Unregister file associations for nvCOMP archive types (requires administrator privileges)");
    parser.addOption(unregisterFileAssocOption);
    
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
    // Handle registration/unregistration commands (no GUI required)
    // ========================================================================
    
#ifdef _WIN32
    // These operations run without showing the GUI and exit immediately
    
    if (parser.isSet(registerContextMenuOption)) {
        std::cout << "Registering Windows Explorer context menu..." << std::endl;
        
        // Check for admin privileges
        if (!ContextMenuManager::isRunningAsAdmin()) {
            std::cerr << "ERROR: Administrator privileges required." << std::endl;
            std::cerr << "Please run this command as administrator:" << std::endl;
            std::cerr << "  1. Right-click nvcomp-gui.exe" << std::endl;
            std::cerr << "  2. Select 'Run as administrator'" << std::endl;
            std::cerr << "  3. Run the command again" << std::endl;
            return 1;
        }
        
        // Get application path
        QString exePath = QCoreApplication::applicationFilePath();
        
        // Register
        bool success = ContextMenuManager::registerContextMenu(exePath, exePath);
        
        if (success) {
            std::cout << "SUCCESS: Windows Explorer context menu registered." << std::endl;
            std::cout << "You can now right-click files/folders in Windows Explorer" << std::endl;
            std::cout << "and select 'Compress with nvCOMP'." << std::endl;
            return 0;
        } else {
            std::cerr << "ERROR: Failed to register context menu." << std::endl;
            std::cerr << "Details: " << ContextMenuManager::getLastError().toStdString() << std::endl;
            return 1;
        }
    }
    
    if (parser.isSet(unregisterContextMenuOption)) {
        std::cout << "Unregistering Windows Explorer context menu..." << std::endl;
        
        // Check if registered
        if (!ContextMenuManager::isRegistered()) {
            std::cout << "Context menu is not currently registered." << std::endl;
            return 0;
        }
        
        // Check for admin privileges
        if (!ContextMenuManager::isRunningAsAdmin()) {
            std::cerr << "ERROR: Administrator privileges required." << std::endl;
            std::cerr << "Please run this command as administrator:" << std::endl;
            std::cerr << "  1. Right-click nvcomp-gui.exe" << std::endl;
            std::cerr << "  2. Select 'Run as administrator'" << std::endl;
            std::cerr << "  3. Run the command again" << std::endl;
            return 1;
        }
        
        // Unregister
        bool success = ContextMenuManager::unregisterContextMenu();
        
        if (success) {
            std::cout << "SUCCESS: Windows Explorer context menu unregistered." << std::endl;
            return 0;
        } else {
            std::cerr << "ERROR: Failed to unregister context menu." << std::endl;
            std::cerr << "Details: " << ContextMenuManager::getLastError().toStdString() << std::endl;
            return 1;
        }
    }
    
    if (parser.isSet(registerFileAssocOption)) {
        std::cout << "Registering file associations..." << std::endl;
        
        // Check for admin privileges
        if (!FileAssociationManager::isRunningAsAdmin()) {
            std::cerr << "ERROR: Administrator privileges required." << std::endl;
            std::cerr << "Please run this command as administrator:" << std::endl;
            std::cerr << "  1. Right-click nvcomp-gui.exe" << std::endl;
            std::cerr << "  2. Select 'Run as administrator'" << std::endl;
            std::cerr << "  3. Run the command again" << std::endl;
            return 1;
        }
        
        // Get application path
        QString exePath = QCoreApplication::applicationFilePath();
        
        // Show what will be registered
        QList<FileTypeInfo> fileTypes = FileAssociationManager::getSupportedFileTypes();
        std::cout << "File types to register:" << std::endl;
        for (const FileTypeInfo &ft : fileTypes) {
            std::cout << "  - " << ft.extension.toStdString() 
                      << " (" << ft.description.toStdString() << ")" << std::endl;
        }
        std::cout << std::endl;
        
        // Register
        bool success = FileAssociationManager::registerAllAssociations(exePath);
        
        if (success) {
            std::cout << "SUCCESS: File associations registered." << std::endl;
            std::cout << "Changes will take effect immediately." << std::endl;
            std::cout << "Features added:" << std::endl;
            std::cout << "  - Double-click compressed files to open in nvCOMP" << std::endl;
            std::cout << "  - Custom icons for each compression algorithm" << std::endl;
            std::cout << "  - Right-click menu: 'Extract here' and 'Extract to folder'" << std::endl;
            return 0;
        } else {
            std::cerr << "ERROR: Failed to register file associations." << std::endl;
            std::cerr << "Details: " << FileAssociationManager::getLastError().toStdString() << std::endl;
            return 1;
        }
    }
    
    if (parser.isSet(unregisterFileAssocOption)) {
        std::cout << "Unregistering file associations..." << std::endl;
        
        // Check if any are registered
        QList<FileTypeInfo> fileTypes = FileAssociationManager::getSupportedFileTypes();
        bool anyRegistered = false;
        for (const FileTypeInfo &ft : fileTypes) {
            if (FileAssociationManager::isAssociated(ft.extension)) {
                anyRegistered = true;
                break;
            }
        }
        
        if (!anyRegistered) {
            std::cout << "No file associations are currently registered." << std::endl;
            return 0;
        }
        
        // Check for admin privileges
        if (!FileAssociationManager::isRunningAsAdmin()) {
            std::cerr << "ERROR: Administrator privileges required." << std::endl;
            std::cerr << "Please run this command as administrator:" << std::endl;
            std::cerr << "  1. Right-click nvcomp-gui.exe" << std::endl;
            std::cerr << "  2. Select 'Run as administrator'" << std::endl;
            std::cerr << "  3. Run the command again" << std::endl;
            return 1;
        }
        
        // Unregister
        bool success = FileAssociationManager::unregisterAllAssociations();
        
        if (success) {
            std::cout << "SUCCESS: File associations unregistered." << std::endl;
            std::cout << "nvCOMP is no longer associated with compressed file types." << std::endl;
            return 0;
        } else {
            std::cerr << "ERROR: Failed to unregister file associations." << std::endl;
            std::cerr << "Details: " << FileAssociationManager::getLastError().toStdString() << std::endl;
            return 1;
        }
    }
#else
    // Non-Windows platforms
    if (parser.isSet(registerContextMenuOption) || 
        parser.isSet(unregisterContextMenuOption) ||
        parser.isSet(registerFileAssocOption) || 
        parser.isSet(unregisterFileAssocOption)) {
        std::cerr << "ERROR: Context menu and file association registration is only supported on Windows." << std::endl;
        return 1;
    }
#endif
    
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

