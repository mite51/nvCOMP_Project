/**
 * @file test_mainwindow.cpp
 * @brief Qt Test suite for MainWindow class
 * 
 * Unit tests for the main window GUI component.
 * Tests window initialization, properties, and basic UI interactions.
 * 
 * Run with: ./test_mainwindow or ctest -R test_mainwindow
 */

#include <QtTest/QtTest>
#include <QApplication>
#include <QMenuBar>
#include <QStatusBar>
#include <QLabel>
#include "mainwindow.h"

/**
 * @class TestMainWindow
 * @brief Test class for MainWindow functionality
 * 
 * Uses Qt Test framework to verify MainWindow behavior.
 * All tests are non-interactive and run automatically.
 */
class TestMainWindow : public QObject
{
    Q_OBJECT

private slots:
    /**
     * @brief Initialize test environment before each test
     */
    void init();
    
    /**
     * @brief Clean up after each test
     */
    void cleanup();
    
    /**
     * @brief Test: MainWindow constructor creates valid object
     */
    void testConstruction();
    
    /**
     * @brief Test: Window has correct initial title
     */
    void testWindowTitle();
    
    /**
     * @brief Test: Window has appropriate initial size
     */
    void testWindowSize();
    
    /**
     * @brief Test: Window has minimum size constraints
     */
    void testMinimumSize();
    
    /**
     * @brief Test: Window is properly initialized
     */
    void testInitialization();
    
    /**
     * @brief Test: Menu bar exists and has expected menus
     */
    void testMenuBar();
    
    /**
     * @brief Test: Status bar exists
     */
    void testStatusBar();
    
    /**
     * @brief Test: Central widget exists and is valid
     */
    void testCentralWidget();
    
    /**
     * @brief Test: About action exists and is properly configured
     */
    void testAboutAction();
    
    /**
     * @brief Test: Exit action exists and is properly configured
     */
    void testExitAction();

private:
    MainWindow *window;  ///< Test window instance
};

void TestMainWindow::init()
{
    // Create a new window for each test
    window = new MainWindow();
}

void TestMainWindow::cleanup()
{
    // Clean up after each test
    delete window;
    window = nullptr;
}

void TestMainWindow::testConstruction()
{
    // Test that window is created successfully
    QVERIFY(window != nullptr);
    QVERIFY(window->isVisible() == false);  // Not shown yet
}

void TestMainWindow::testWindowTitle()
{
    // Test window title is set correctly
    QString title = window->windowTitle();
    QVERIFY(!title.isEmpty());
    QVERIFY(title.contains("nvCOMP", Qt::CaseInsensitive));
}

void TestMainWindow::testWindowSize()
{
    // Test initial window size
    QSize size = window->size();
    
    // Default size should be 800x600 (as set in mainwindow.cpp)
    QCOMPARE(size.width(), 800);
    QCOMPARE(size.height(), 600);
}

void TestMainWindow::testMinimumSize()
{
    // Test minimum size constraints
    QSize minSize = window->minimumSize();
    
    // Minimum size should be 640x480
    QCOMPARE(minSize.width(), 640);
    QCOMPARE(minSize.height(), 480);
    
    // Verify we cannot resize below minimum
    window->resize(400, 300);
    QSize actualSize = window->size();
    QVERIFY(actualSize.width() >= 640);
    QVERIFY(actualSize.height() >= 480);
}

void TestMainWindow::testInitialization()
{
    // Test that window reports as initialized
    QVERIFY(window->isInitialized());
}

void TestMainWindow::testMenuBar()
{
    // Test menu bar exists
    QMenuBar *menuBar = window->menuBar();
    QVERIFY(menuBar != nullptr);
    
    // Test expected menus exist
    QList<QAction*> actions = menuBar->actions();
    QVERIFY(actions.size() >= 2);  // At least File and Help menus
    
    // Find File and Help menus
    bool hasFileMenu = false;
    bool hasHelpMenu = false;
    
    for (QAction *action : actions) {
        QString text = action->text().remove('&');  // Remove mnemonics
        if (text.contains("File", Qt::CaseInsensitive)) {
            hasFileMenu = true;
        }
        if (text.contains("Help", Qt::CaseInsensitive)) {
            hasHelpMenu = true;
        }
    }
    
    QVERIFY(hasFileMenu);
    QVERIFY(hasHelpMenu);
}

void TestMainWindow::testStatusBar()
{
    // Test status bar exists
    QStatusBar *statusBar = window->statusBar();
    QVERIFY(statusBar != nullptr);
}

void TestMainWindow::testCentralWidget()
{
    // Test central widget exists
    QWidget *central = window->centralWidget();
    QVERIFY(central != nullptr);
    
    // Test central widget has layout
    QLayout *layout = central->layout();
    QVERIFY(layout != nullptr);
}

void TestMainWindow::testAboutAction()
{
    // Find About action
    QAction *aboutAction = window->findChild<QAction*>("actionAbout");
    QVERIFY(aboutAction != nullptr);
    
    // Test properties
    QVERIFY(!aboutAction->text().isEmpty());
    QVERIFY(aboutAction->text().contains("About", Qt::CaseInsensitive));
    
    // Test shortcut (F1)
    QKeySequence shortcut = aboutAction->shortcut();
    QCOMPARE(shortcut, QKeySequence(Qt::Key_F1));
}

void TestMainWindow::testExitAction()
{
    // Find Exit action
    QAction *exitAction = window->findChild<QAction*>("actionExit");
    QVERIFY(exitAction != nullptr);
    
    // Test properties
    QVERIFY(!exitAction->text().isEmpty());
    QVERIFY(exitAction->text().contains("Exit", Qt::CaseInsensitive));
    
    // Test shortcut (Ctrl+Q)
    QKeySequence shortcut = exitAction->shortcut();
    QCOMPARE(shortcut, QKeySequence(Qt::CTRL | Qt::Key_Q));
}

// ============================================================================
// Test Application Main
// ============================================================================

/**
 * @brief Main entry point for GUI tests
 * 
 * Qt Test requires a QApplication instance for GUI testing.
 * This creates the application and runs all tests.
 */
int main(int argc, char *argv[])
{
    // Create QApplication for GUI testing
    // Use QApplication::setSetuidAllowed(true) if running as root (Linux only)
    QApplication app(argc, argv);
    
    // Suppress window display during tests (headless testing)
    app.setAttribute(Qt::AA_Use96Dpi, true);
    
    // Create test object and run
    TestMainWindow test;
    
    // Run tests with Qt Test framework
    // Returns 0 if all tests pass, non-zero otherwise
    return QTest::qExec(&test, argc, argv);
}

// Include the generated MOC file for Qt's meta-object compiler
#include "test_mainwindow.moc"

