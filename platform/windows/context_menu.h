/**
 * @file context_menu.h
 * @brief Windows Explorer context menu integration for nvCOMP
 * 
 * Provides registry-based context menu integration allowing users to
 * compress files and folders directly from Windows Explorer.
 */

#ifndef CONTEXT_MENU_H
#define CONTEXT_MENU_H

#include <QString>
#include <QStringList>

/**
 * @class ContextMenuManager
 * @brief Manages Windows registry integration for context menus
 * 
 * Registers and unregisters context menu entries in the Windows registry
 * for both files and directories. Supports cascading submenus with
 * different compression algorithm options.
 */
class ContextMenuManager
{
public:
    /**
     * @brief Registers context menu entries in Windows registry
     * @param exePath Full path to the nvcomp-gui.exe executable
     * @param iconPath Full path to the application icon (optional)
     * @return true if registration succeeded, false otherwise
     * 
     * Creates registry entries under:
     * - HKEY_CLASSES_ROOT\\*\\shell\\nvCOMP (for files)
     * - HKEY_CLASSES_ROOT\\Directory\\shell\\nvCOMP (for folders)
     * - HKEY_CLASSES_ROOT\\Directory\\Background\\shell\\nvCOMP (for folder background)
     * 
     * Requires administrator privileges on Windows.
     */
    static bool registerContextMenu(const QString &exePath, const QString &iconPath = QString());
    
    /**
     * @brief Unregisters context menu entries from Windows registry
     * @return true if unregistration succeeded, false otherwise
     * 
     * Removes all registry entries created by registerContextMenu().
     * Requires administrator privileges on Windows.
     */
    static bool unregisterContextMenu();
    
    /**
     * @brief Checks if context menu is currently registered
     * @return true if registered, false otherwise
     */
    static bool isRegistered();
    
    /**
     * @brief Checks if the current process has administrator privileges
     * @return true if running as administrator, false otherwise
     * 
     * Context menu registration/unregistration requires admin rights.
     */
    static bool isRunningAsAdmin();
    
    /**
     * @brief Gets the last error message from registry operations
     * @return QString containing the error description
     */
    static QString getLastError();
    
private:
    static QString s_lastError;  ///< Stores the last error message
    
    /**
     * @brief Sets a registry string value
     * @param keyPath Full registry path (e.g., "HKEY_CLASSES_ROOT\\*\\shell\\nvCOMP")
     * @param valueName Name of the value to set (empty for default value)
     * @param valueData String data to write
     * @return true if successful, false otherwise
     */
    static bool setRegistryValue(const QString &keyPath, const QString &valueName, const QString &valueData);
    
    /**
     * @brief Deletes a registry key and all subkeys
     * @param keyPath Full registry path to delete
     * @return true if successful, false otherwise
     */
    static bool deleteRegistryKey(const QString &keyPath);
    
    /**
     * @brief Checks if a registry key exists
     * @param keyPath Full registry path to check
     * @return true if exists, false otherwise
     */
    static bool registryKeyExists(const QString &keyPath);
};

#endif // CONTEXT_MENU_H

