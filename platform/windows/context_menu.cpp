/**
 * @file context_menu.cpp
 * @brief Implementation of Windows Explorer context menu integration
 */

#include "context_menu.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <shlwapi.h>  // For SHDeleteKeyW
#include <QSettings>
#include <QFileInfo>
#include <QDir>
#include <QDebug>

QString ContextMenuManager::s_lastError;

bool ContextMenuManager::registerContextMenu(const QString &exePath, const QString &iconPath)
{
    s_lastError.clear();
    
    // NOTE: This creates a cascading context menu using MUIVerb + SubCommands
    // This should work on both Windows 10 and Windows 11
    // On Windows 11, it may still appear in "Show more options" due to system restrictions
    // Full Windows 11 modern menu integration would require a COM-based shell extension
    
    // Validate exe path
    QFileInfo exeInfo(exePath);
    if (!exeInfo.exists() || !exeInfo.isFile()) {
        s_lastError = QString("Executable not found: %1").arg(exePath);
        return false;
    }
    
    // Use native path separators
    QString nativeExePath = QDir::toNativeSeparators(exeInfo.absoluteFilePath());
    QString nativeIconPath = iconPath.isEmpty() ? nativeExePath : QDir::toNativeSeparators(iconPath);
    
    bool success = true;
    
    // ========================================================================
    // Register for files (HKEY_CLASSES_ROOT\*\shell\nvCOMP)
    // ========================================================================
    
    QString fileMenuKey = "HKEY_CLASSES_ROOT\\*\\shell\\nvCOMP";
    
    // Main menu item - Use MUIVerb for proper cascading menu
    setRegistryValue(fileMenuKey, "MUIVerb", "Compress with nvCOMP");
    
    // Set icon (always set it, using exe path with icon index)
    // Format: "path\to\exe,0" means first icon resource in exe
    QString iconPathWithIndex = QString("\"%1\",0").arg(nativeIconPath);
    setRegistryValue(fileMenuKey, "Icon", iconPathWithIndex);
    
    // Enable cascading menu (submenus)
    setRegistryValue(fileMenuKey, "SubCommands", "");
    
    // Set position hint (appears near "Open with")
    setRegistryValue(fileMenuKey, "Position", "Middle");
    
    // Windows 11 compatibility: Add to "Send To" style registration
    // This helps Windows 11 recognize it as a legitimate handler
    setRegistryValue(fileMenuKey, "AppliesTo", "*");
    
    // Add extended verb flag (allows Shift+Right-click to always work)
    // Note: We want it visible by default, so we DON'T set Extended flag
    // But we document this for reference
    
    // Submenu: Compress with LZ4
    QString lz4Key = fileMenuKey + "\\shell\\LZ4";
    setRegistryValue(lz4Key, "MUIVerb", "Compress here (LZ4)");
    setRegistryValue(lz4Key, "Icon", iconPathWithIndex);
    setRegistryValue(lz4Key + "\\command", "", 
        QString("\"%1\" --compress --algorithm lz4 \"%2\"").arg(nativeExePath).arg("%1"));
    
    // Submenu: Compress with Zstd
    QString zstdKey = fileMenuKey + "\\shell\\Zstd";
    setRegistryValue(zstdKey, "MUIVerb", "Compress here (Zstd)");
    setRegistryValue(zstdKey, "Icon", iconPathWithIndex);
    setRegistryValue(zstdKey + "\\command", "", 
        QString("\"%1\" --compress --algorithm zstd \"%2\"").arg(nativeExePath).arg("%1"));
    
    // Submenu: Compress with Snappy
    QString snappyKey = fileMenuKey + "\\shell\\Snappy";
    setRegistryValue(snappyKey, "MUIVerb", "Compress here (Snappy)");
    setRegistryValue(snappyKey, "Icon", iconPathWithIndex);
    setRegistryValue(snappyKey + "\\command", "", 
        QString("\"%1\" --compress --algorithm snappy \"%2\"").arg(nativeExePath).arg("%1"));
    
    // Submenu: Choose algorithm (opens GUI)
    QString chooseKey = fileMenuKey + "\\shell\\Choose";
    setRegistryValue(chooseKey, "MUIVerb", "Choose algorithm...");
    setRegistryValue(chooseKey, "Icon", iconPathWithIndex);
    setRegistryValue(chooseKey + "\\command", "", 
        QString("\"%1\" --add-file \"%2\"").arg(nativeExePath).arg("%1"));
    
    // ========================================================================
    // Register for folders (HKEY_CLASSES_ROOT\Directory\shell\nvCOMP)
    // ========================================================================
    
    QString folderMenuKey = "HKEY_CLASSES_ROOT\\Directory\\shell\\nvCOMP";
    
    // Main menu item for folders - Use MUIVerb for proper cascading menu
    setRegistryValue(folderMenuKey, "MUIVerb", "Compress with nvCOMP");
    
    // Set icon (always set it with icon index)
    setRegistryValue(folderMenuKey, "Icon", iconPathWithIndex);
    
    // Enable cascading menu (submenus)
    setRegistryValue(folderMenuKey, "SubCommands", "");
    
    // Set position hint
    setRegistryValue(folderMenuKey, "Position", "Middle");
    
    // Windows 11 compatibility
    setRegistryValue(folderMenuKey, "AppliesTo", "*");
    
    // Submenu: Compress folder with LZ4
    QString folderLz4Key = folderMenuKey + "\\shell\\LZ4";
    setRegistryValue(folderLz4Key, "MUIVerb", "Compress folder (LZ4)");
    setRegistryValue(folderLz4Key, "Icon", iconPathWithIndex);
    setRegistryValue(folderLz4Key + "\\command", "", 
        QString("\"%1\" --compress --algorithm lz4 \"%2\"").arg(nativeExePath).arg("%1"));
    
    // Submenu: Compress folder with Zstd
    QString folderZstdKey = folderMenuKey + "\\shell\\Zstd";
    setRegistryValue(folderZstdKey, "MUIVerb", "Compress folder (Zstd)");
    setRegistryValue(folderZstdKey, "Icon", iconPathWithIndex);
    setRegistryValue(folderZstdKey + "\\command", "", 
        QString("\"%1\" --compress --algorithm zstd \"%2\"").arg(nativeExePath).arg("%1"));
    
    // Submenu: Compress folder with Snappy
    QString folderSnappyKey = folderMenuKey + "\\shell\\Snappy";
    setRegistryValue(folderSnappyKey, "MUIVerb", "Compress folder (Snappy)");
    setRegistryValue(folderSnappyKey, "Icon", iconPathWithIndex);
    setRegistryValue(folderSnappyKey + "\\command", "", 
        QString("\"%1\" --compress --algorithm snappy \"%2\"").arg(nativeExePath).arg("%1"));
    
    // Submenu: Choose algorithm (opens GUI)
    QString folderChooseKey = folderMenuKey + "\\shell\\Choose";
    setRegistryValue(folderChooseKey, "MUIVerb", "Choose algorithm...");
    setRegistryValue(folderChooseKey, "Icon", iconPathWithIndex);
    setRegistryValue(folderChooseKey + "\\command", "", 
        QString("\"%1\" --add-file \"%2\"").arg(nativeExePath).arg("%1"));
    
    // ========================================================================
    // Register for folder background (right-click in empty folder area)
    // ========================================================================
    
    QString bgMenuKey = "HKEY_CLASSES_ROOT\\Directory\\Background\\shell\\nvCOMP";
    
    setRegistryValue(bgMenuKey, "", "Compress to archive...");
    if (!iconPath.isEmpty()) {
        setRegistryValue(bgMenuKey, "Icon", nativeIconPath);
    }
    setRegistryValue(bgMenuKey + "\\command", "", 
        QString("\"%1\"").arg(nativeExePath));
    
    // Notify shell of changes
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, NULL, NULL);
    
    if (!success) {
        s_lastError = "Some registry operations failed. Context menu may be partially registered.";
    }
    
    return success;
}

bool ContextMenuManager::unregisterContextMenu()
{
    s_lastError.clear();
    
    bool success = true;
    
    // Delete registry keys
    if (!deleteRegistryKey("HKEY_CLASSES_ROOT\\*\\shell\\nvCOMP")) {
        success = false;
    }
    
    if (!deleteRegistryKey("HKEY_CLASSES_ROOT\\Directory\\shell\\nvCOMP")) {
        success = false;
    }
    
    if (!deleteRegistryKey("HKEY_CLASSES_ROOT\\Directory\\Background\\shell\\nvCOMP")) {
        success = false;
    }
    
    // Notify shell of changes
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, NULL, NULL);
    
    if (!success) {
        s_lastError = "Some registry keys could not be deleted. Context menu may be partially unregistered.";
    }
    
    return success;
}

bool ContextMenuManager::isRegistered()
{
    // Check if any of the main keys exist
    return registryKeyExists("HKEY_CLASSES_ROOT\\*\\shell\\nvCOMP") ||
           registryKeyExists("HKEY_CLASSES_ROOT\\Directory\\shell\\nvCOMP");
}

bool ContextMenuManager::isRunningAsAdmin()
{
#ifdef _WIN32
    BOOL isAdmin = FALSE;
    PSID administratorsGroup = NULL;
    SID_IDENTIFIER_AUTHORITY ntAuthority = SECURITY_NT_AUTHORITY;
    
    // Create a SID for the Administrators group
    if (AllocateAndInitializeSid(&ntAuthority, 2,
                                  SECURITY_BUILTIN_DOMAIN_RID,
                                  DOMAIN_ALIAS_RID_ADMINS,
                                  0, 0, 0, 0, 0, 0,
                                  &administratorsGroup))
    {
        // Check if the current user is a member of the Administrators group
        if (!CheckTokenMembership(NULL, administratorsGroup, &isAdmin))
        {
            isAdmin = FALSE;
        }
        FreeSid(administratorsGroup);
    }
    
    return isAdmin == TRUE;
#else
    return false;
#endif
}

QString ContextMenuManager::getLastError()
{
    return s_lastError;
}

bool ContextMenuManager::setRegistryValue(const QString &keyPath, const QString &valueName, const QString &valueData)
{
    QSettings settings(keyPath, QSettings::NativeFormat);
    
    // QSettings requires "." to set the default (unnamed) value
    // Empty string doesn't work correctly with QSettings
    QString actualValueName = valueName.isEmpty() ? "." : valueName;
    
    settings.setValue(actualValueName, valueData);
    settings.sync();
    
    if (settings.status() != QSettings::NoError) {
        s_lastError = QString("Failed to write registry: %1\\%2").arg(keyPath).arg(valueName);
        qWarning() << s_lastError;
        return false;
    }
    
    return true;
}

bool ContextMenuManager::deleteRegistryKey(const QString &keyPath)
{
    // QSettings doesn't directly support deleting entire keys with subkeys
    // We need to use Windows API for this
    
#ifdef _WIN32
    // Convert Qt registry path to Windows registry path
    QString winPath = keyPath;
    winPath.replace("HKEY_CLASSES_ROOT\\", "");
    
    HKEY hKey = HKEY_CLASSES_ROOT;
    
    // Convert QString to wchar_t for Windows API
    std::wstring wPath = winPath.toStdWString();
    
    // Delete the key and all subkeys recursively
    LONG result = SHDeleteKeyW(hKey, wPath.c_str());
    
    if (result != ERROR_SUCCESS && result != ERROR_FILE_NOT_FOUND) {
        s_lastError = QString("Failed to delete registry key: %1 (Error code: %2)")
                          .arg(keyPath).arg(result);
        qWarning() << s_lastError;
        return false;
    }
    
    return true;
#else
    return false;
#endif
}

bool ContextMenuManager::registryKeyExists(const QString &keyPath)
{
    QSettings settings(keyPath, QSettings::NativeFormat);
    
    // Check if the key contains any values or subkeys
    QStringList keys = settings.childKeys();
    QStringList groups = settings.childGroups();
    
    // Check for default value using "." (QSettings convention)
    QString defaultValue = settings.value(".").toString();
    
    // If default value exists or has children, the key exists
    return !defaultValue.isEmpty() || !keys.isEmpty() || !groups.isEmpty();
}

#else // Not Windows

// Stub implementations for non-Windows platforms
bool ContextMenuManager::registerContextMenu(const QString &, const QString &)
{
    s_lastError = "Context menu registration is only supported on Windows";
    return false;
}

bool ContextMenuManager::unregisterContextMenu()
{
    s_lastError = "Context menu unregistration is only supported on Windows";
    return false;
}

bool ContextMenuManager::isRegistered()
{
    return false;
}

bool ContextMenuManager::isRunningAsAdmin()
{
    return false;
}

QString ContextMenuManager::getLastError()
{
    return s_lastError;
}

bool ContextMenuManager::setRegistryValue(const QString &, const QString &, const QString &)
{
    return false;
}

bool ContextMenuManager::deleteRegistryKey(const QString &)
{
    return false;
}

bool ContextMenuManager::registryKeyExists(const QString &)
{
    return false;
}

#endif // _WIN32

