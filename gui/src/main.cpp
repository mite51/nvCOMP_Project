/**
 * @file main.cpp
 * @brief nvCOMP GUI Application Entry Point
 * 
 * Qt-based graphical interface for NVIDIA nvCOMP compression tool.
 * Provides drag-and-drop compression/decompression with GPU acceleration.
 */

#include <QApplication>
#include <QStyleFactory>
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
    
    // Create and show main window
    MainWindow window;
    window.show();
    
    // Enter Qt event loop
    return app.exec();
}

