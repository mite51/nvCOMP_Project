/**
 * @file mainwindow.cpp
 * @brief Implementation of MainWindow class
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_initialized(false)
{
    ui->setupUi(this);
    setupUi();
    setupConnections();
    m_initialized = true;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setupUi()
{
    // Set window properties
    setWindowTitle("nvCOMP - GPU Accelerated Compression");
    resize(800, 600);
    
    // Set minimum size to ensure usability
    setMinimumSize(640, 480);
    
    // Center window on screen (will be enhanced in future tasks)
    // For now, let Qt position it
}

void MainWindow::setupConnections()
{
    // Connect menu actions (defined in .ui file)
    // About action
    if (ui->actionAbout) {
        connect(ui->actionAbout, &QAction::triggered, 
                this, &MainWindow::onAboutTriggered);
    }
    
    // Exit action
    if (ui->actionExit) {
        connect(ui->actionExit, &QAction::triggered, 
                this, &MainWindow::onExitTriggered);
    }
}

QString MainWindow::windowTitle() const
{
    return QMainWindow::windowTitle();
}

bool MainWindow::isInitialized() const
{
    return m_initialized;
}

void MainWindow::onAboutTriggered()
{
    QMessageBox::about(this, 
        tr("About nvCOMP"),
        tr("<h3>nvCOMP GUI v1.0.0</h3>"
           "<p>GPU-accelerated compression tool using NVIDIA nvCOMP</p>"
           "<p><b>Supported Algorithms:</b></p>"
           "<ul>"
           "<li>LZ4 - Fast compression</li>"
           "<li>Snappy - Very fast compression</li>"
           "<li>Zstd - Best compression ratios</li>"
           "<li>GDeflate - GPU-optimized DEFLATE</li>"
           "<li>ANS - Asymmetric Numeral Systems</li>"
           "<li>Bitcomp - Lossless numerical compression</li>"
           "</ul>"
           "<p>Copyright © 2024</p>"));
}

void MainWindow::onExitTriggered()
{
    // Clean exit (future tasks will add cleanup for workers/threads)
    QApplication::quit();
}

