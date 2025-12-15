/**
 * @file settings_dialog.cpp
 * @brief Implementation of SettingsDialog class
 */

#include "settings_dialog.h"
#include "ui_settings_dialog.h"
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QFileInfo>

SettingsDialog::SettingsDialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::SettingsDialog)
    , m_settings("nvCOMP", "nvCOMP GUI")
{
    ui->setupUi(this);
    setupUi();
    setupConnections();
    loadSettings();
}

SettingsDialog::~SettingsDialog()
{
    delete ui;
}

void SettingsDialog::setupUi()
{
    // Set window properties
    setWindowTitle("Settings");
    setModal(true);
    resize(600, 500);
    
    // Set minimum size
    setMinimumSize(500, 400);
    
    // Manually set userData for algorithm combo box (Qt Designer userData sometimes doesn't work)
    ui->comboBoxDefaultAlgorithm->setItemData(0, "LZ4");
    ui->comboBoxDefaultAlgorithm->setItemData(1, "Snappy");
    ui->comboBoxDefaultAlgorithm->setItemData(2, "Zstd");
    ui->comboBoxDefaultAlgorithm->setItemData(3, "GDeflate");
    ui->comboBoxDefaultAlgorithm->setItemData(4, "ANS");
    ui->comboBoxDefaultAlgorithm->setItemData(5, "Bitcomp");
}

void SettingsDialog::setupConnections()
{
    // Dialog buttons
    connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &SettingsDialog::onAccepted);
    connect(ui->buttonBox, &QDialogButtonBox::rejected, this, &SettingsDialog::onRejected);
    
    // Apply button (if exists in button box)
    QPushButton *applyButton = ui->buttonBox->button(QDialogButtonBox::Apply);
    if (applyButton) {
        connect(applyButton, &QPushButton::clicked, this, &SettingsDialog::onApplyClicked);
    }
    
    // Restore Defaults button
    connect(ui->buttonRestoreDefaults, &QPushButton::clicked, 
            this, &SettingsDialog::onRestoreDefaultsClicked);
    
    // Validation connections
    connect(ui->lineEditOutputTemplate, &QLineEdit::textChanged,
            this, &SettingsDialog::onOutputTemplateChanged);
    connect(ui->spinBoxDefaultVolumeSize, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SettingsDialog::onVolumeSizeChanged);
}

void SettingsDialog::loadSettings()
{
    // Tab 1: Compression
    QString algorithm = m_settings.value("compression/defaultAlgorithm", "LZ4").toString();
    int algorithmIndex = ui->comboBoxDefaultAlgorithm->findData(algorithm);
    if (algorithmIndex >= 0) {
        ui->comboBoxDefaultAlgorithm->setCurrentIndex(algorithmIndex);
    }
    
    ui->spinBoxDefaultVolumeSize->setValue(
        m_settings.value("compression/defaultVolumeSize", 2560).toInt());
    
    ui->checkBoxDefaultEnableVolumes->setChecked(
        m_settings.value("compression/defaultEnableVolumes", true).toBool());
    
    ui->lineEditOutputTemplate->setText(
        m_settings.value("compression/outputTemplate", "{filename}.nvcomp").toString());
    
    // Tab 2: Performance
    bool preferGpu = m_settings.value("performance/preferGpu", true).toBool();
    if (preferGpu) {
        ui->radioButtonPreferGpu->setChecked(true);
    } else {
        ui->radioButtonPreferCpu->setChecked(true);
    }
    
    ui->sliderVramLimit->setValue(
        m_settings.value("performance/vramLimit", 80).toInt());
    
    ui->spinBoxThreadCount->setValue(
        m_settings.value("performance/threadCount", 4).toInt());
    
    ui->spinBoxChunkSize->setValue(
        m_settings.value("performance/chunkSize", 128).toInt());
    
    // Tab 3: Interface
    QString language = m_settings.value("interface/language", "English").toString();
    int languageIndex = ui->comboBoxLanguage->findText(language);
    if (languageIndex >= 0) {
        ui->comboBoxLanguage->setCurrentIndex(languageIndex);
    }
    
    QString theme = m_settings.value("interface/theme", "System").toString();
    int themeIndex = ui->comboBoxTheme->findText(theme);
    if (themeIndex >= 0) {
        ui->comboBoxTheme->setCurrentIndex(themeIndex);
    }
    
    ui->checkBoxConfirmOverwrite->setChecked(
        m_settings.value("interface/confirmOverwrite", true).toBool());
    
    ui->checkBoxShowStatistics->setChecked(
        m_settings.value("interface/showStatistics", true).toBool());
    
    // Tab 4: Integration
    ui->checkBoxEnableContextMenu->setChecked(
        m_settings.value("integration/enableContextMenu", false).toBool());
    
    ui->checkBoxEnableFileAssociations->setChecked(
        m_settings.value("integration/enableFileAssociations", false).toBool());
    
    ui->checkBoxStartWithSystem->setChecked(
        m_settings.value("integration/startWithSystem", false).toBool());
}

void SettingsDialog::saveSettings()
{
    // Tab 1: Compression
    m_settings.setValue("compression/defaultAlgorithm", 
                       ui->comboBoxDefaultAlgorithm->currentData().toString());
    
    m_settings.setValue("compression/defaultVolumeSize", 
                       ui->spinBoxDefaultVolumeSize->value());
    m_settings.setValue("compression/defaultEnableVolumes", 
                       ui->checkBoxDefaultEnableVolumes->isChecked());
    m_settings.setValue("compression/outputTemplate", 
                       ui->lineEditOutputTemplate->text());
    
    // Tab 2: Performance
    m_settings.setValue("performance/preferGpu", 
                       ui->radioButtonPreferGpu->isChecked());
    m_settings.setValue("performance/vramLimit", 
                       ui->sliderVramLimit->value());
    m_settings.setValue("performance/threadCount", 
                       ui->spinBoxThreadCount->value());
    m_settings.setValue("performance/chunkSize", 
                       ui->spinBoxChunkSize->value());
    
    // Tab 3: Interface
    m_settings.setValue("interface/language", 
                       ui->comboBoxLanguage->currentText());
    m_settings.setValue("interface/theme", 
                       ui->comboBoxTheme->currentText());
    m_settings.setValue("interface/confirmOverwrite", 
                       ui->checkBoxConfirmOverwrite->isChecked());
    m_settings.setValue("interface/showStatistics", 
                       ui->checkBoxShowStatistics->isChecked());
    
    // Tab 4: Integration
    m_settings.setValue("integration/enableContextMenu", 
                       ui->checkBoxEnableContextMenu->isChecked());
    m_settings.setValue("integration/enableFileAssociations", 
                       ui->checkBoxEnableFileAssociations->isChecked());
    m_settings.setValue("integration/startWithSystem", 
                       ui->checkBoxStartWithSystem->isChecked());
    
    // Sync to disk
    m_settings.sync();
}

void SettingsDialog::restoreDefaults()
{
    // Tab 1: Compression
    ui->comboBoxDefaultAlgorithm->setCurrentIndex(0);  // LZ4
    ui->spinBoxDefaultVolumeSize->setValue(2560);  // 2.5 GB
    ui->checkBoxDefaultEnableVolumes->setChecked(true);
    ui->lineEditOutputTemplate->setText("{filename}.nvcomp");
    
    // Tab 2: Performance
    ui->radioButtonPreferGpu->setChecked(true);
    ui->sliderVramLimit->setValue(80);
    ui->spinBoxThreadCount->setValue(4);
    ui->spinBoxChunkSize->setValue(128);
    
    // Tab 3: Interface
    ui->comboBoxLanguage->setCurrentIndex(0);  // English
    ui->comboBoxTheme->setCurrentIndex(0);  // System
    ui->checkBoxConfirmOverwrite->setChecked(true);
    ui->checkBoxShowStatistics->setChecked(true);
    
    // Tab 4: Integration
    ui->checkBoxEnableContextMenu->setChecked(false);
    ui->checkBoxEnableFileAssociations->setChecked(false);
    ui->checkBoxStartWithSystem->setChecked(false);
}

// Getters - Compression
QString SettingsDialog::getDefaultAlgorithm() const
{
    return m_settings.value("compression/defaultAlgorithm", "LZ4").toString();
}

int SettingsDialog::getDefaultVolumeSize() const
{
    return m_settings.value("compression/defaultVolumeSize", 2560).toInt();
}

bool SettingsDialog::getDefaultEnableVolumes() const
{
    return m_settings.value("compression/defaultEnableVolumes", true).toBool();
}

QString SettingsDialog::getOutputPathTemplate() const
{
    return m_settings.value("compression/outputTemplate", "{filename}.nvcomp").toString();
}

// Getters - Performance
bool SettingsDialog::getPreferGpu() const
{
    return m_settings.value("performance/preferGpu", true).toBool();
}

int SettingsDialog::getVramLimit() const
{
    return m_settings.value("performance/vramLimit", 80).toInt();
}

int SettingsDialog::getThreadCount() const
{
    return m_settings.value("performance/threadCount", 4).toInt();
}

int SettingsDialog::getChunkSize() const
{
    return m_settings.value("performance/chunkSize", 128).toInt();
}

// Getters - Interface
QString SettingsDialog::getLanguage() const
{
    return m_settings.value("interface/language", "English").toString();
}

QString SettingsDialog::getTheme() const
{
    return m_settings.value("interface/theme", "System").toString();
}

bool SettingsDialog::getConfirmOverwrite() const
{
    return m_settings.value("interface/confirmOverwrite", true).toBool();
}

bool SettingsDialog::getShowStatistics() const
{
    return m_settings.value("interface/showStatistics", true).toBool();
}

// Getters - Integration
bool SettingsDialog::getEnableContextMenu() const
{
    return m_settings.value("integration/enableContextMenu", false).toBool();
}

bool SettingsDialog::getEnableFileAssociations() const
{
    return m_settings.value("integration/enableFileAssociations", false).toBool();
}

bool SettingsDialog::getStartWithSystem() const
{
    return m_settings.value("integration/startWithSystem", false).toBool();
}

void SettingsDialog::onAccepted()
{
    if (validateSettings()) {
        saveSettings();
        emit settingsApplied();
        accept();
    }
}

void SettingsDialog::onApplyClicked()
{
    if (validateSettings()) {
        saveSettings();
        emit settingsApplied();
        QMessageBox::information(this, tr("Settings Applied"),
                                tr("Settings have been saved successfully."));
    }
}

void SettingsDialog::onRejected()
{
    // Don't save changes
    reject();
}

void SettingsDialog::onRestoreDefaultsClicked()
{
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        tr("Restore Defaults"),
        tr("Are you sure you want to restore all settings to their default values?"),
        QMessageBox::Yes | QMessageBox::No
    );
    
    if (reply == QMessageBox::Yes) {
        restoreDefaults();
        QMessageBox::information(this, tr("Defaults Restored"),
                                tr("All settings have been restored to default values.\n"
                                   "Click OK or Apply to save these changes."));
    }
}

void SettingsDialog::onOutputTemplateChanged(const QString &text)
{
    // Validate template contains {filename}
    if (!text.contains("{filename}")) {
        ui->lineEditOutputTemplate->setStyleSheet("QLineEdit { background-color: #fff3cd; }");
        ui->lineEditOutputTemplate->setToolTip("Warning: Template should contain {filename}");
    } else {
        ui->lineEditOutputTemplate->setStyleSheet("");
        ui->lineEditOutputTemplate->setToolTip("Output path template. Use {filename} for original name.");
    }
}

void SettingsDialog::onVolumeSizeChanged(int value)
{
    // Warn if volume size is very small or very large
    if (value < 10) {
        ui->spinBoxDefaultVolumeSize->setStyleSheet("QSpinBox { background-color: #fff3cd; }");
        ui->spinBoxDefaultVolumeSize->setToolTip("Warning: Very small volume size may create many files");
    } else if (value > 5000) {
        ui->spinBoxDefaultVolumeSize->setStyleSheet("QSpinBox { background-color: #fff3cd; }");
        ui->spinBoxDefaultVolumeSize->setToolTip("Warning: Large volume size may exceed GPU memory");
    } else {
        ui->spinBoxDefaultVolumeSize->setStyleSheet("");
        ui->spinBoxDefaultVolumeSize->setToolTip("Default size for volume splitting (in MB)");
    }
}

bool SettingsDialog::validateSettings()
{
    // Validate output template
    QString outputTemplate = ui->lineEditOutputTemplate->text().trimmed();
    if (outputTemplate.isEmpty()) {
        QMessageBox::warning(this, tr("Invalid Setting"),
                           tr("Output path template cannot be empty."));
        ui->tabWidget->setCurrentIndex(0);  // Switch to Compression tab
        ui->lineEditOutputTemplate->setFocus();
        return false;
    }
    
    // Validate volume size
    int volumeSize = ui->spinBoxDefaultVolumeSize->value();
    if (volumeSize < 1 || volumeSize > 10000) {
        QMessageBox::warning(this, tr("Invalid Setting"),
                           tr("Volume size must be between 1 and 10000 MB."));
        ui->tabWidget->setCurrentIndex(0);  // Switch to Compression tab
        ui->spinBoxDefaultVolumeSize->setFocus();
        return false;
    }
    
    // Validate thread count
    int threadCount = ui->spinBoxThreadCount->value();
    if (threadCount < 1 || threadCount > 64) {
        QMessageBox::warning(this, tr("Invalid Setting"),
                           tr("Thread count must be between 1 and 64."));
        ui->tabWidget->setCurrentIndex(1);  // Switch to Performance tab
        ui->spinBoxThreadCount->setFocus();
        return false;
    }
    
    // Validate chunk size
    int chunkSize = ui->spinBoxChunkSize->value();
    if (chunkSize < 1 || chunkSize > 1024) {
        QMessageBox::warning(this, tr("Invalid Setting"),
                           tr("Chunk size must be between 1 and 1024 MB."));
        ui->tabWidget->setCurrentIndex(1);  // Switch to Performance tab
        ui->spinBoxChunkSize->setFocus();
        return false;
    }
    
    return true;
}

void SettingsDialog::applySettings()
{
    // This method can be used to apply settings to the application
    // without closing the dialog. Currently handled by saveSettings().
    saveSettings();
}

QVariant SettingsDialog::getDefaultValue(const QString &key, const QVariant &defaultValue) const
{
    return m_settings.value(key, defaultValue);
}

