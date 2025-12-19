/**
 * @file desktop_integration.h
 * @brief Linux desktop environment integration for nvCOMP
 * 
 * Provides functionality to integrate nvCOMP with Linux desktop environments
 * following freedesktop.org standards (XDG specifications).
 * Supports GNOME, KDE, XFCE, and other compliant desktop environments.
 */

#ifndef DESKTOP_INTEGRATION_H
#define DESKTOP_INTEGRATION_H

#include <QString>
#include <QStringList>
#include <QProcess>

/**
 * @class DesktopIntegration
 * @brief Manages Linux desktop integration for nvCOMP application
 * 
 * This class handles:
 * - .desktop file generation and installation
 * - MIME type definitions and registration
 * - Icon installation (multiple sizes)
 * - XDG database updates
 * - Default application registration
 * 
 * Supports both system-wide (/usr/share) and user-only (~/.local/share) installation.
 */
class DesktopIntegration
{
public:
    /**
     * @brief Installation scope
     */
    enum InstallScope {
        SystemWide,  ///< Install to /usr/share (requires root)
        UserOnly     ///< Install to ~/.local/share (no root needed)
    };
    
    /**
     * @brief Integration status
     */
    struct Status {
        bool desktopFileInstalled;
        bool mimeTypesInstalled;
        bool iconsInstalled;
        bool isDefaultApplication;
        QString installPath;
        QString error;
    };
    
    /**
     * @brief Constructs a DesktopIntegration instance
     * @param executablePath Path to the nvcomp-gui executable
     * @param scope Installation scope (system-wide or user-only)
     */
    explicit DesktopIntegration(const QString &executablePath, InstallScope scope = UserOnly);
    
    /**
     * @brief Installs desktop integration files
     * @return true if successful, false otherwise
     * 
     * Installs:
     * - .desktop file
     * - MIME type definitions
     * - Application icons
     * - Updates XDG databases
     */
    bool install();
    
    /**
     * @brief Removes desktop integration files
     * @return true if successful, false otherwise
     */
    bool uninstall();
    
    /**
     * @brief Checks if desktop integration is installed
     * @return true if installed, false otherwise
     */
    bool isInstalled() const;
    
    /**
     * @brief Gets detailed integration status
     * @return Status structure with installation details
     */
    Status getStatus() const;
    
    /**
     * @brief Sets as default application for supported MIME types
     * @return true if successful, false otherwise
     */
    bool setAsDefaultApplication();
    
    /**
     * @brief Gets last error message
     * @return Error message string
     */
    QString lastError() const { return m_lastError; }
    
    /**
     * @brief Gets installation scope
     * @return Current installation scope
     */
    InstallScope scope() const { return m_scope; }
    
    /**
     * @brief Sets installation scope
     * @param scope New installation scope
     */
    void setScope(InstallScope scope) { m_scope = scope; }

private:
    // Installation methods
    bool installDesktopFile();
    bool installMimeTypes();
    bool installIcons();
    bool updateDesktopDatabase();
    bool updateMimeDatabase();
    bool updateIconCache();
    
    // Uninstallation methods
    bool uninstallDesktopFile();
    bool uninstallMimeTypes();
    bool uninstallIcons();
    
    // Helper methods
    QString getDesktopFilePath() const;
    QString getMimePackagePath() const;
    QString getIconBasePath() const;
    QString getApplicationsPath() const;
    
    QString generateDesktopFileContent() const;
    QString generateMimeTypeXml() const;
    
    bool ensureDirectoryExists(const QString &path);
    bool writeFile(const QString &path, const QString &content);
    bool removeFile(const QString &path);
    bool runCommand(const QString &program, const QStringList &arguments);
    
    // MIME type helpers
    QStringList getSupportedMimeTypes() const;
    QString getMimeTypeForExtension(const QString &extension) const;
    
    // Icon helpers
    bool installIconForSize(int size);
    QByteArray generateIconData(int size) const;
    
private:
    QString m_executablePath;
    InstallScope m_scope;
    mutable QString m_lastError;
};

#endif // DESKTOP_INTEGRATION_H


