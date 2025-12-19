/**
 * @file desktop_integration.cpp
 * @brief Implementation of Linux desktop integration
 */

#include "desktop_integration.h"
#include <QFile>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QCoreApplication>
#include <QDebug>
#include <QImage>
#include <QPainter>
#include <QIcon>
#include <QBuffer>

DesktopIntegration::DesktopIntegration(const QString &executablePath, InstallScope scope)
    : m_executablePath(executablePath)
    , m_scope(scope)
{
}

bool DesktopIntegration::install()
{
    m_lastError.clear();
    
    // Install components
    if (!installDesktopFile()) {
        return false;
    }
    
    if (!installMimeTypes()) {
        return false;
    }
    
    if (!installIcons()) {
        return false;
    }
    
    // Update databases
    updateDesktopDatabase();
    updateMimeDatabase();
    updateIconCache();
    
    return true;
}

bool DesktopIntegration::uninstall()
{
    m_lastError.clear();
    
    bool success = true;
    
    if (!uninstallDesktopFile()) {
        success = false;
    }
    
    if (!uninstallMimeTypes()) {
        success = false;
    }
    
    if (!uninstallIcons()) {
        success = false;
    }
    
    // Update databases
    updateDesktopDatabase();
    updateMimeDatabase();
    updateIconCache();
    
    return success;
}

bool DesktopIntegration::isInstalled() const
{
    QString desktopFile = getDesktopFilePath();
    return QFile::exists(desktopFile);
}

DesktopIntegration::Status DesktopIntegration::getStatus() const
{
    Status status;
    status.desktopFileInstalled = QFile::exists(getDesktopFilePath());
    status.mimeTypesInstalled = QFile::exists(getMimePackagePath());
    status.iconsInstalled = QFile::exists(getIconBasePath() + "/48x48/apps/nvcomp.png");
    status.isDefaultApplication = false; // TODO: Check xdg-mime default
    status.installPath = getApplicationsPath();
    status.error = m_lastError;
    
    return status;
}

bool DesktopIntegration::setAsDefaultApplication()
{
    m_lastError.clear();
    
    QStringList mimeTypes = getSupportedMimeTypes();
    bool allSuccess = true;
    
    for (const QString &mimeType : mimeTypes) {
        QStringList args;
        args << "default" << "nvcomp.desktop" << mimeType;
        
        if (!runCommand("xdg-mime", args)) {
            allSuccess = false;
        }
    }
    
    return allSuccess;
}

// Installation methods

bool DesktopIntegration::installDesktopFile()
{
    QString content = generateDesktopFileContent();
    QString filePath = getDesktopFilePath();
    
    if (!ensureDirectoryExists(QFileInfo(filePath).absolutePath())) {
        m_lastError = QString("Failed to create directory: %1").arg(QFileInfo(filePath).absolutePath());
        return false;
    }
    
    if (!writeFile(filePath, content)) {
        m_lastError = QString("Failed to write desktop file: %1").arg(filePath);
        return false;
    }
    
    // Make desktop file executable (required by some systems)
    QFile file(filePath);
    file.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner |
                        QFile::ReadGroup | QFile::ExeGroup |
                        QFile::ReadOther | QFile::ExeOther);
    
    return true;
}

bool DesktopIntegration::installMimeTypes()
{
    QString content = generateMimeTypeXml();
    QString filePath = getMimePackagePath();
    
    if (!ensureDirectoryExists(QFileInfo(filePath).absolutePath())) {
        m_lastError = QString("Failed to create directory: %1").arg(QFileInfo(filePath).absolutePath());
        return false;
    }
    
    if (!writeFile(filePath, content)) {
        m_lastError = QString("Failed to write MIME type file: %1").arg(filePath);
        return false;
    }
    
    return true;
}

bool DesktopIntegration::installIcons()
{
    // Ensure the icon theme index exists
    QString iconBasePath = getIconBasePath();
    QString indexThemePath = iconBasePath + "/index.theme";
    
    if (!QFile::exists(indexThemePath)) {
        // Create index.theme for hicolor icon theme
        QString indexContent = 
            "[Icon Theme]\n"
            "Name=Hicolor\n"
            "Comment=Fallback icon theme\n"
            "Hidden=true\n"
            "Directories=16x16/apps,32x32/apps,48x48/apps,64x64/apps,128x128/apps,256x256/apps\n"
            "\n"
            "[16x16/apps]\n"
            "Size=16\n"
            "Context=Applications\n"
            "Type=Threshold\n"
            "\n"
            "[32x32/apps]\n"
            "Size=32\n"
            "Context=Applications\n"
            "Type=Threshold\n"
            "\n"
            "[48x48/apps]\n"
            "Size=48\n"
            "Context=Applications\n"
            "Type=Threshold\n"
            "\n"
            "[64x64/apps]\n"
            "Size=64\n"
            "Context=Applications\n"
            "Type=Threshold\n"
            "\n"
            "[128x128/apps]\n"
            "Size=128\n"
            "Context=Applications\n"
            "Type=Threshold\n"
            "\n"
            "[256x256/apps]\n"
            "Size=256\n"
            "Context=Applications\n"
            "Type=Threshold\n";
        
        if (!ensureDirectoryExists(iconBasePath)) {
            m_lastError = QString("Failed to create icon directory: %1").arg(iconBasePath);
            return false;
        }
        
        if (!writeFile(indexThemePath, indexContent)) {
            m_lastError = QString("Failed to create index.theme: %1").arg(indexThemePath);
            return false;
        }
    }
    
    // Install icons in multiple sizes
    QList<int> sizes = {16, 32, 48, 64, 128, 256};
    bool allSuccess = true;
    
    for (int size : sizes) {
        if (!installIconForSize(size)) {
            allSuccess = false;
        }
    }
    
    return allSuccess;
}

bool DesktopIntegration::updateDesktopDatabase()
{
    QString path = getApplicationsPath();
    return runCommand("update-desktop-database", QStringList() << path);
}

bool DesktopIntegration::updateMimeDatabase()
{
    QString basePath = (m_scope == SystemWide) ? "/usr/share" : 
                       QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
    QString mimePath = basePath + "/mime";
    
    return runCommand("update-mime-database", QStringList() << mimePath);
}

bool DesktopIntegration::updateIconCache()
{
    QString iconPath = getIconBasePath();
    
    // gtk-update-icon-cache is optional, don't fail if it's not available
    runCommand("gtk-update-icon-cache", QStringList() << "-f" << "-t" << iconPath);
    
    return true; // Always return true as this is optional
}

// Uninstallation methods

bool DesktopIntegration::uninstallDesktopFile()
{
    return removeFile(getDesktopFilePath());
}

bool DesktopIntegration::uninstallMimeTypes()
{
    return removeFile(getMimePackagePath());
}

bool DesktopIntegration::uninstallIcons()
{
    QList<int> sizes = {16, 32, 48, 64, 128, 256};
    bool allSuccess = true;
    
    for (int size : sizes) {
        QString iconPath = QString("%1/%2x%2/apps/nvcomp.png")
                               .arg(getIconBasePath())
                               .arg(size);
        if (!removeFile(iconPath)) {
            allSuccess = false;
        }
    }
    
    return allSuccess;
}

// Helper methods - Paths

QString DesktopIntegration::getDesktopFilePath() const
{
    return getApplicationsPath() + "/nvcomp.desktop";
}

QString DesktopIntegration::getMimePackagePath() const
{
    QString basePath = (m_scope == SystemWide) ? "/usr/share" : 
                       QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
    return basePath + "/mime/packages/nvcomp.xml";
}

QString DesktopIntegration::getIconBasePath() const
{
    QString basePath = (m_scope == SystemWide) ? "/usr/share" : 
                       QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
    return basePath + "/icons/hicolor";
}

QString DesktopIntegration::getApplicationsPath() const
{
    if (m_scope == SystemWide) {
        return "/usr/share/applications";
    } else {
        return QStandardPaths::writableLocation(QStandardPaths::ApplicationsLocation);
    }
}

// Content generation

QString DesktopIntegration::generateDesktopFileContent() const
{
    QString appName = QCoreApplication::applicationName();
    if (appName.isEmpty()) {
        appName = "nvCOMP";
    }
    
    QStringList mimeTypes = getSupportedMimeTypes();
    QString mimeTypeStr = mimeTypes.join(";") + ";";
    
    QString content;
    content += "[Desktop Entry]\n";
    content += "Version=1.0\n";
    content += "Type=Application\n";
    content += "Name=nvCOMP\n";
    content += "GenericName=GPU-Accelerated Compression\n";
    content += "Comment=Compress and decompress files using NVIDIA GPU acceleration\n";
    content += QString("Exec=%1 %f\n").arg(m_executablePath);
    content += "Icon=nvcomp\n";
    content += "Terminal=false\n";
    content += "Categories=Utility;Archiving;Compression;Qt;\n";
    content += QString("MimeType=%1\n").arg(mimeTypeStr);
    content += "Keywords=compress;decompress;archive;lz4;zstd;snappy;gpu;cuda;\n";
    content += "StartupNotify=true\n";
    content += "StartupWMClass=nvcomp\n";
    content += "\n";
    
    // Desktop Actions
    content += "[Desktop Action Compress]\n";
    content += "Name=Compress Files\n";
    content += QString("Exec=%1 -c\n").arg(m_executablePath);
    content += "\n";
    
    content += "[Desktop Action Decompress]\n";
    content += "Name=Decompress Archive\n";
    content += QString("Exec=%1 -d\n").arg(m_executablePath);
    content += "\n";
    
    return content;
}

QString DesktopIntegration::generateMimeTypeXml() const
{
    QString xml;
    xml += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    xml += "<mime-info xmlns=\"http://www.freedesktop.org/standards/shared-mime-info\">\n";
    xml += "\n";
    
    // LZ4 MIME type
    xml += "  <mime-type type=\"application/x-lz4\">\n";
    xml += "    <comment>LZ4 compressed archive</comment>\n";
    xml += "    <glob pattern=\"*.lz4\"/>\n";
    xml += "    <glob pattern=\"*.vol*.lz4\"/>\n";
    xml += "    <magic priority=\"50\">\n";
    xml += "      <match type=\"string\" offset=\"0\" value=\"NVBC\"/>\n";
    xml += "    </magic>\n";
    xml += "    <icon name=\"nvcomp\"/>\n";
    xml += "  </mime-type>\n";
    xml += "\n";
    
    // Zstd MIME type
    xml += "  <mime-type type=\"application/x-zstd\">\n";
    xml += "    <comment>Zstd compressed archive</comment>\n";
    xml += "    <glob pattern=\"*.zstd\"/>\n";
    xml += "    <glob pattern=\"*.zst\"/>\n";
    xml += "    <glob pattern=\"*.vol*.zstd\"/>\n";
    xml += "    <magic priority=\"50\">\n";
    xml += "      <match type=\"string\" offset=\"0\" value=\"NVBC\"/>\n";
    xml += "    </magic>\n";
    xml += "    <icon name=\"nvcomp\"/>\n";
    xml += "  </mime-type>\n";
    xml += "\n";
    
    // Snappy MIME type
    xml += "  <mime-type type=\"application/x-snappy\">\n";
    xml += "    <comment>Snappy compressed archive</comment>\n";
    xml += "    <glob pattern=\"*.snappy\"/>\n";
    xml += "    <glob pattern=\"*.vol*.snappy\"/>\n";
    xml += "    <magic priority=\"50\">\n";
    xml += "      <match type=\"string\" offset=\"0\" value=\"NVBC\"/>\n";
    xml += "    </magic>\n";
    xml += "    <icon name=\"nvcomp\"/>\n";
    xml += "  </mime-type>\n";
    xml += "\n";
    
    // nvCOMP generic MIME type
    xml += "  <mime-type type=\"application/x-nvcomp\">\n";
    xml += "    <comment>nvCOMP compressed archive</comment>\n";
    xml += "    <glob pattern=\"*.nvcomp\"/>\n";
    xml += "    <glob pattern=\"*.gdeflate\"/>\n";
    xml += "    <glob pattern=\"*.ans\"/>\n";
    xml += "    <glob pattern=\"*.bitcomp\"/>\n";
    xml += "    <icon name=\"nvcomp\"/>\n";
    xml += "  </mime-type>\n";
    xml += "\n";
    
    xml += "</mime-info>\n";
    
    return xml;
}

// File operations

bool DesktopIntegration::ensureDirectoryExists(const QString &path)
{
    QDir dir;
    if (!dir.exists(path)) {
        return dir.mkpath(path);
    }
    return true;
}

bool DesktopIntegration::writeFile(const QString &path, const QString &content)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream out(&file);
    out << content;
    file.close();
    
    return true;
}

bool DesktopIntegration::removeFile(const QString &path)
{
    QFile file(path);
    if (file.exists()) {
        return file.remove();
    }
    return true; // File doesn't exist, consider it removed
}

bool DesktopIntegration::runCommand(const QString &program, const QStringList &arguments)
{
    QProcess process;
    process.start(program, arguments);
    
    if (!process.waitForStarted()) {
        return false;
    }
    
    if (!process.waitForFinished(30000)) { // 30 second timeout
        process.kill();
        return false;
    }
    
    return process.exitCode() == 0;
}

// MIME type helpers

QStringList DesktopIntegration::getSupportedMimeTypes() const
{
    return QStringList{
        "application/x-lz4",
        "application/x-zstd",
        "application/x-snappy",
        "application/x-nvcomp"
    };
}

QString DesktopIntegration::getMimeTypeForExtension(const QString &extension) const
{
    if (extension == "lz4") {
        return "application/x-lz4";
    } else if (extension == "zstd" || extension == "zst") {
        return "application/x-zstd";
    } else if (extension == "snappy") {
        return "application/x-snappy";
    } else if (extension == "nvcomp" || extension == "gdeflate" || 
               extension == "ans" || extension == "bitcomp") {
        return "application/x-nvcomp";
    }
    return QString();
}

// Icon helpers

bool DesktopIntegration::installIconForSize(int size)
{
    QString iconDir = QString("%1/%2x%2/apps").arg(getIconBasePath()).arg(size);
    QString iconPath = iconDir + "/nvcomp.png";
    
    if (!ensureDirectoryExists(iconDir)) {
        return false;
    }
    
    // Generate a simple icon (green square with "nv" text)
    // In production, you'd use actual icon files
    QImage image(size, size, QImage::Format_ARGB32);
    image.fill(Qt::transparent);
    
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Background gradient (green to dark green)
    QLinearGradient gradient(0, 0, 0, size);
    gradient.setColorAt(0, QColor(118, 185, 0));  // NVIDIA green
    gradient.setColorAt(1, QColor(82, 129, 0));
    
    painter.setBrush(gradient);
    painter.setPen(Qt::NoPen);
    painter.drawRoundedRect(0, 0, size, size, size * 0.15, size * 0.15);
    
    // Draw "nv" text
    if (size >= 32) {
        QFont font;
        font.setPixelSize(size * 0.4);
        font.setBold(true);
        painter.setFont(font);
        painter.setPen(Qt::white);
        painter.drawText(QRect(0, 0, size, size), Qt::AlignCenter, "nv");
    }
    
    painter.end();
    
    // Save the icon
    if (!image.save(iconPath, "PNG")) {
        return false;
    }
    
    return true;
}

QByteArray DesktopIntegration::generateIconData(int size) const
{
    QImage image(size, size, QImage::Format_ARGB32);
    image.fill(Qt::transparent);
    
    // Simple icon generation
    QPainter painter(&image);
    painter.fillRect(0, 0, size, size, QColor(118, 185, 0)); // NVIDIA green
    painter.end();
    
    QByteArray data;
    QBuffer buffer(&data);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "PNG");
    
    return data;
}

