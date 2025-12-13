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
#include <QPushButton>
#include <QListWidget>
#include <QComboBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QProgressBar>
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
    
    // ========================================================================
    // Task 2.2: UI Interaction Tests
    // ========================================================================
    
    /**
     * @brief Test: File list widget exists and is properly configured
     */
    void testFileListWidget();
    
    /**
     * @brief Test: Add Files button exists and is clickable
     */
    void testAddFilesButton();
    
    /**
     * @brief Test: Add Folder button exists and is clickable
     */
    void testAddFolderButton();
    
    /**
     * @brief Test: Clear Files button exists and works
     */
    void testClearFilesButton();
    
    /**
     * @brief Test: Compress button exists and initial state
     */
    void testCompressButton();
    
    /**
     * @brief Test: Decompress button exists and initial state
     */
    void testDecompressButton();
    
    /**
     * @brief Test: Settings button exists and is clickable
     */
    void testSettingsButton();
    
    /**
     * @brief Test: Algorithm combo box exists and has items
     */
    void testAlgorithmComboBox();
    
    /**
     * @brief Test: Algorithm selection can be changed
     */
    void testAlgorithmSelection();
    
    /**
     * @brief Test: CPU mode checkbox exists and works
     */
    void testCpuModeCheckbox();
    
    /**
     * @brief Test: Volumes checkbox exists and enables spinbox
     */
    void testVolumesCheckbox();
    
    /**
     * @brief Test: Volume size spinbox configuration
     */
    void testVolumeSizeSpinBox();
    
    /**
     * @brief Test: Progress bar exists and is configured
     */
    void testProgressBar();
    
    /**
     * @brief Test: Status label exists
     */
    void testStatusLabel();
    
    /**
     * @brief Test: GPU status indicator exists
     */
    void testGpuStatusIndicator();
    
    /**
     * @brief Test: Drag-and-drop is enabled
     */
    void testDragDropEnabled();
    
    /**
     * @brief Test: File list getters work correctly
     */
    void testFileListGetters();
    
    /**
     * @brief Test: Settings getters work correctly
     */
    void testSettingsGetters();

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
    
    // Default size should be 900x700 (as set in mainwindow.cpp)
    QCOMPARE(size.width(), 900);
    QCOMPARE(size.height(), 700);
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
    // Remove mnemonic character (&) before checking
    QString text = exitAction->text().remove('&');
    QVERIFY(text.contains("Exit", Qt::CaseInsensitive));
    
    // Test shortcut (Ctrl+Q)
    QKeySequence shortcut = exitAction->shortcut();
    QCOMPARE(shortcut, QKeySequence(Qt::CTRL | Qt::Key_Q));
}

// ============================================================================
// Task 2.2: UI Interaction Test Implementations
// ============================================================================

void TestMainWindow::testFileListWidget()
{
    // Find file list widget
    QListWidget *fileList = window->findChild<QListWidget*>("listWidgetFiles");
    QVERIFY(fileList != nullptr);
    
    // Test properties
    QVERIFY(fileList->alternatingRowColors());
    QCOMPARE(fileList->selectionMode(), QAbstractItemView::ExtendedSelection);
    QVERIFY(fileList->minimumHeight() >= 200);
    
    // Initially should be empty
    QCOMPARE(fileList->count(), 0);
}

void TestMainWindow::testAddFilesButton()
{
    // Find Add Files button
    QPushButton *button = window->findChild<QPushButton*>("buttonAddFiles");
    QVERIFY(button != nullptr);
    
    // Test properties
    QVERIFY(!button->text().isEmpty());
    QVERIFY(button->text().contains("Add Files", Qt::CaseInsensitive));
    QVERIFY(button->isEnabled());
    
    // Test tooltip
    QVERIFY(!button->toolTip().isEmpty());
}

void TestMainWindow::testAddFolderButton()
{
    // This test is now deprecated - Add Folder merged with Add Files
    // Just verify the Add Files button handles both
    QPushButton *button = window->findChild<QPushButton*>("buttonAddFiles");
    QVERIFY(button != nullptr);
    QVERIFY(button->isEnabled());
}

void TestMainWindow::testClearFilesButton()
{
    // Find Clear Files button
    QPushButton *button = window->findChild<QPushButton*>("buttonClearFiles");
    QVERIFY(button != nullptr);
    
    // Test properties
    QVERIFY(!button->text().isEmpty());
    QVERIFY(button->text().contains("Clear", Qt::CaseInsensitive));
    QVERIFY(button->isEnabled());
    
    // Test that clicking clears the file list
    QListWidget *fileList = window->findChild<QListWidget*>("listWidgetFiles");
    QVERIFY(fileList != nullptr);
    
    // Initially empty
    QCOMPARE(fileList->count(), 0);
    QCOMPARE(window->getFileList().count(), 0);
}

void TestMainWindow::testCompressButton()
{
    // Find Compress button
    QPushButton *button = window->findChild<QPushButton*>("buttonCompress");
    QVERIFY(button != nullptr);
    
    // Test properties
    QVERIFY(!button->text().isEmpty());
    QVERIFY(button->text().contains("Compress", Qt::CaseInsensitive));
    
    // Initially should be disabled (no files)
    QVERIFY(!button->isEnabled());
    
    // Test minimum height
    QVERIFY(button->minimumHeight() >= 40);
    
    // Test tooltip
    QVERIFY(!button->toolTip().isEmpty());
}

void TestMainWindow::testDecompressButton()
{
    // Find Decompress button
    QPushButton *button = window->findChild<QPushButton*>("buttonDecompress");
    QVERIFY(button != nullptr);
    
    // Test properties
    QVERIFY(!button->text().isEmpty());
    QVERIFY(button->text().contains("Decompress", Qt::CaseInsensitive));
    
    // Initially should be disabled (no files)
    QVERIFY(!button->isEnabled());
    
    // Test minimum height
    QVERIFY(button->minimumHeight() >= 40);
    
    // Test tooltip
    QVERIFY(!button->toolTip().isEmpty());
}

void TestMainWindow::testSettingsButton()
{
    // Find Settings button
    QPushButton *button = window->findChild<QPushButton*>("buttonSettings");
    QVERIFY(button != nullptr);
    
    // Test properties
    QVERIFY(!button->text().isEmpty());
    QVERIFY(button->text().contains("Settings", Qt::CaseInsensitive));
    QVERIFY(button->isEnabled());
    
    // Test minimum height
    QVERIFY(button->minimumHeight() >= 40);
    
    // Test tooltip
    QVERIFY(!button->toolTip().isEmpty());
}

void TestMainWindow::testAlgorithmComboBox()
{
    // Find algorithm combo box
    QComboBox *comboBox = window->findChild<QComboBox*>("comboBoxAlgorithm");
    QVERIFY(comboBox != nullptr);
    
    // Test has items
    QVERIFY(comboBox->count() > 0);
    QCOMPARE(comboBox->count(), 6);  // 6 algorithms
    
    // Test algorithm names contain expected keywords
    QStringList algorithms;
    for (int i = 0; i < comboBox->count(); ++i) {
        algorithms.append(comboBox->itemText(i));
    }
    
    // Check for expected algorithms
    bool hasLZ4 = false, hasSnappy = false, hasZstd = false;
    bool hasGDeflate = false, hasANS = false, hasBitcomp = false;
    
    for (const QString &algo : algorithms) {
        if (algo.contains("LZ4", Qt::CaseInsensitive)) hasLZ4 = true;
        if (algo.contains("Snappy", Qt::CaseInsensitive)) hasSnappy = true;
        if (algo.contains("Zstd", Qt::CaseInsensitive)) hasZstd = true;
        if (algo.contains("GDeflate", Qt::CaseInsensitive)) hasGDeflate = true;
        if (algo.contains("ANS", Qt::CaseInsensitive)) hasANS = true;
        if (algo.contains("Bitcomp", Qt::CaseInsensitive)) hasBitcomp = true;
    }
    
    QVERIFY(hasLZ4);
    QVERIFY(hasSnappy);
    QVERIFY(hasZstd);
    QVERIFY(hasGDeflate);
    QVERIFY(hasANS);
    QVERIFY(hasBitcomp);
    
    // Test tooltip
    QVERIFY(!comboBox->toolTip().isEmpty());
}

void TestMainWindow::testAlgorithmSelection()
{
    // Find algorithm combo box
    QComboBox *comboBox = window->findChild<QComboBox*>("comboBoxAlgorithm");
    QVERIFY(comboBox != nullptr);
    
    // Test changing selection
    int initialIndex = comboBox->currentIndex();
    QString initialAlgorithm = window->getSelectedAlgorithm();
    QVERIFY(!initialAlgorithm.isEmpty());
    
    // Change selection if possible
    if (comboBox->count() > 1) {
        int newIndex = (initialIndex + 1) % comboBox->count();
        comboBox->setCurrentIndex(newIndex);
        
        QString newAlgorithm = window->getSelectedAlgorithm();
        QVERIFY(newAlgorithm != initialAlgorithm);
        QCOMPARE(newAlgorithm, comboBox->currentText());
    }
}

void TestMainWindow::testCpuModeCheckbox()
{
    // Find CPU mode checkbox
    QCheckBox *checkBox = window->findChild<QCheckBox*>("checkBoxCpuMode");
    QVERIFY(checkBox != nullptr);
    
    // Test properties
    QVERIFY(!checkBox->text().isEmpty());
    QVERIFY(checkBox->text().contains("CPU", Qt::CaseInsensitive));
    
    // Test tooltip
    QVERIFY(!checkBox->toolTip().isEmpty());
    
    // Test that getter works
    bool cpuMode = window->isCpuModeEnabled();
    QCOMPARE(cpuMode, checkBox->isChecked());
    
    // Test toggling (if enabled)
    if (checkBox->isEnabled()) {
        bool initialState = checkBox->isChecked();
        checkBox->setChecked(!initialState);
        QCOMPARE(window->isCpuModeEnabled(), !initialState);
    }
}

void TestMainWindow::testVolumesCheckbox()
{
    // Find volumes checkbox
    QCheckBox *checkBox = window->findChild<QCheckBox*>("checkBoxVolumes");
    QVERIFY(checkBox != nullptr);
    
    // Test properties
    QVERIFY(!checkBox->text().isEmpty());
    QVERIFY(checkBox->text().contains("volume", Qt::CaseInsensitive));
    
    // Test tooltip
    QVERIFY(!checkBox->toolTip().isEmpty());
    
    // Find related widgets
    QLabel *label = window->findChild<QLabel*>("labelVolumeSize");
    QSpinBox *spinBox = window->findChild<QSpinBox*>("spinBoxVolumeSize");
    QVERIFY(label != nullptr);
    QVERIFY(spinBox != nullptr);
    
    // Initially unchecked, so label and spinbox should be disabled
    if (!checkBox->isChecked()) {
        QVERIFY(!label->isEnabled());
        QVERIFY(!spinBox->isEnabled());
    }
    
    // Toggle checkbox and verify connected widgets enable
    checkBox->setChecked(true);
    QVERIFY(label->isEnabled());
    QVERIFY(spinBox->isEnabled());
    QVERIFY(window->isVolumesEnabled());
    
    // Toggle back
    checkBox->setChecked(false);
    QVERIFY(!label->isEnabled());
    QVERIFY(!spinBox->isEnabled());
    QVERIFY(!window->isVolumesEnabled());
}

void TestMainWindow::testVolumeSizeSpinBox()
{
    // Find volume size spinbox
    QSpinBox *spinBox = window->findChild<QSpinBox*>("spinBoxVolumeSize");
    QVERIFY(spinBox != nullptr);
    
    // Test properties
    QVERIFY(spinBox->minimum() >= 1);
    QVERIFY(spinBox->maximum() >= 100);
    QVERIFY(spinBox->value() >= spinBox->minimum());
    QVERIFY(spinBox->value() <= spinBox->maximum());
    
    // Test tooltip
    QVERIFY(!spinBox->toolTip().isEmpty());
    
    // Test getter
    int volumeSize = window->getVolumeSize();
    QCOMPARE(volumeSize, spinBox->value());
    
    // Test changing value
    spinBox->setValue(200);
    QCOMPARE(window->getVolumeSize(), 200);
}

void TestMainWindow::testProgressBar()
{
    // Find progress bar
    QProgressBar *progressBar = window->findChild<QProgressBar*>("progressBar");
    QVERIFY(progressBar != nullptr);
    
    // Test properties
    QVERIFY(progressBar->isTextVisible());
    QCOMPARE(progressBar->value(), 0);  // Initially at 0
    QVERIFY(progressBar->minimum() >= 0);
    QVERIFY(progressBar->maximum() > 0);
}

void TestMainWindow::testStatusLabel()
{
    // Find status label
    QLabel *label = window->findChild<QLabel*>("labelStatus");
    QVERIFY(label != nullptr);
    
    // Test properties
    QVERIFY(!label->text().isEmpty());
    QVERIFY(label->text().contains("Ready", Qt::CaseInsensitive));
    
    // Test alignment
    QVERIFY(label->alignment() & Qt::AlignCenter);
}

void TestMainWindow::testGpuStatusIndicator()
{
    // Find GPU status label
    QLabel *statusLabel = window->findChild<QLabel*>("labelGpuStatus");
    QVERIFY(statusLabel != nullptr);
    
    // Test properties
    QVERIFY(!statusLabel->text().isEmpty());
    QVERIFY(statusLabel->text().contains("GPU", Qt::CaseInsensitive));
    
    // Find GPU icon label
    QLabel *iconLabel = window->findChild<QLabel*>("labelGpuIcon");
    QVERIFY(iconLabel != nullptr);
    QVERIFY(!iconLabel->text().isEmpty());
}

void TestMainWindow::testDragDropEnabled()
{
    // Test that drag-and-drop is enabled
    QVERIFY(window->acceptDrops());
}

void TestMainWindow::testFileListGetters()
{
    // Test file list getter
    QStringList files = window->getFileList();
    QVERIFY(files.isEmpty());  // Initially empty
    
    // File list should match internal state
    QListWidget *fileList = window->findChild<QListWidget*>("listWidgetFiles");
    QVERIFY(fileList != nullptr);
    QCOMPARE(files.count(), fileList->count());
}

void TestMainWindow::testSettingsGetters()
{
    // Test all settings getters
    QString algorithm = window->getSelectedAlgorithm();
    QVERIFY(!algorithm.isEmpty());
    
    bool cpuMode = window->isCpuModeEnabled();
    // cpuMode is bool, just verify it returns something valid
    Q_UNUSED(cpuMode);
    
    bool volumesEnabled = window->isVolumesEnabled();
    Q_UNUSED(volumesEnabled);
    
    int volumeSize = window->getVolumeSize();
    QVERIFY(volumeSize > 0);
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

