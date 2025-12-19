/**
 * @file settings_dialog.h
 * @brief Settings dialog for nvCOMP GUI application
 * 
 * Provides user preferences configuration with persistence using QSettings.
 * Includes 4 tabs: Compression, Performance, Interface, and Integration.
 */

#ifndef SETTINGS_DIALOG_H
#define SETTINGS_DIALOG_H

#include <QDialog>
#include <QSettings>

// Forward declarations for platform-specific classes
#ifdef Q_OS_LINUX
class DesktopIntegration;
#endif

QT_BEGIN_NAMESPACE
namespace Ui { class SettingsDialog; }
QT_END_NAMESPACE

/**
 * @class SettingsDialog
 * @brief Dialog for managing application settings
 * 
 * Provides a tabbed interface for configuring compression settings,
 * performance options, interface preferences, and system integration.
 * All settings are persisted using QSettings.
 */
class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * @brief Constructs the settings dialog
     * @param parent Parent widget (usually MainWindow)
     */
    explicit SettingsDialog(QWidget *parent = nullptr);
    
    /**
     * @brief Destroys the settings dialog
     */
    ~SettingsDialog();
    
    // Compression settings getters
    QString getDefaultAlgorithm() const;
    int getDefaultVolumeSize() const;
    bool getDefaultEnableVolumes() const;
    QString getOutputPathTemplate() const;
    
    // Performance settings getters
    bool getPreferGpu() const;
    int getVramLimit() const;
    int getThreadCount() const;
    int getChunkSize() const;
    
    // Interface settings getters
    QString getLanguage() const;
    QString getTheme() const;
    bool getConfirmOverwrite() const;
    bool getShowStatistics() const;
    
    // Integration settings getters
    bool getEnableContextMenu() const;
    bool getEnableFileAssociations() const;
    bool getStartWithSystem() const;
    
    /**
     * @brief Loads settings from QSettings
     */
    void loadSettings();
    
    /**
     * @brief Saves settings to QSettings
     */
    void saveSettings();
    
    /**
     * @brief Restores all settings to default values
     */
    void restoreDefaults();

signals:
    /**
     * @brief Emitted when settings are applied
     * 
     * Connected widgets should update their state based on new settings.
     */
    void settingsApplied();

private slots:
    /**
     * @brief Handles OK button click
     * 
     * Saves settings and closes the dialog.
     */
    void onAccepted();
    
    /**
     * @brief Handles Apply button click
     * 
     * Saves settings without closing the dialog.
     */
    void onApplyClicked();
    
    /**
     * @brief Handles Cancel button click
     * 
     * Closes the dialog without saving changes.
     */
    void onRejected();
    
    /**
     * @brief Handles Restore Defaults button click
     * 
     * Resets all settings to default values.
     */
    void onRestoreDefaultsClicked();
    
    /**
     * @brief Validates the output path template
     * @param text Current template text
     */
    void onOutputTemplateChanged(const QString &text);
    
    /**
     * @brief Validates the volume size
     * @param value Current volume size value
     */
    void onVolumeSizeChanged(int value);
    
#ifdef Q_OS_LINUX
    /**
     * @brief Handles Linux desktop integration checkbox toggle
     * @param checked Whether integration should be enabled
     */
    void onLinuxDesktopIntegrationToggled(bool checked);
#endif

private:
    Ui::SettingsDialog *ui;  ///< Qt Designer generated UI
    QSettings m_settings;    ///< Settings storage
    
#ifdef Q_OS_LINUX
    DesktopIntegration *m_desktopIntegration;  ///< Linux desktop integration helper
#endif
    
    /**
     * @brief Sets up UI components and connections
     */
    void setupUi();
    
    /**
     * @brief Connects signals and slots
     */
    void setupConnections();
    
    /**
     * @brief Applies UI values to application settings
     */
    void applySettings();
    
    /**
     * @brief Validates all settings
     * @return true if all settings are valid
     */
    bool validateSettings();
    
    /**
     * @brief Gets default value for a setting
     * @param key Setting key
     * @param defaultValue Default value if key doesn't exist
     * @return Setting value or default
     */
    QVariant getDefaultValue(const QString &key, const QVariant &defaultValue) const;
};

#endif // SETTINGS_DIALOG_H

