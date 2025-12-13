/**
 * @file mainwindow.h
 * @brief Main window class for nvCOMP GUI application
 * 
 * Provides the primary user interface for compression/decompression operations.
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QStringList>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

/**
 * @class MainWindow
 * @brief Main application window
 * 
 * Handles user interactions, file operations, and displays compression progress.
 * Future tasks will add compression workers, GPU monitoring, and settings dialogs.
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    /**
     * @brief Constructs the main window
     * @param parent Parent widget (nullptr for top-level window)
     */
    explicit MainWindow(QWidget *parent = nullptr);
    
    /**
     * @brief Destroys the main window and cleans up resources
     */
    ~MainWindow();
    
    /**
     * @brief Gets the window title
     * @return Current window title string
     */
    QString windowTitle() const;
    
    /**
     * @brief Checks if the main window is properly initialized
     * @return true if window is valid and ready
     */
    bool isInitialized() const;

private:
    Ui::MainWindow *ui;  ///< Qt Designer generated UI
    bool m_initialized;  ///< Initialization state flag
    
    /**
     * @brief Initializes UI components and connections
     */
    void setupUi();
    
    /**
     * @brief Connects signals and slots for UI interactions
     */
    void setupConnections();
    
private slots:
    /**
     * @brief Handles About dialog display
     * Shows application information and version
     */
    void onAboutTriggered();
    
    /**
     * @brief Handles application exit
     * Performs cleanup before closing
     */
    void onExitTriggered();
};

#endif // MAINWINDOW_H

