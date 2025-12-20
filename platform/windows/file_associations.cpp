/**
 * @file file_associations.cpp
 * @brief Implementation of Windows file association management for nvCOMP
 */

#include "file_associations.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <QString>
#include <QDir>
#include <QDebug>

// Initialize static members
QString FileAssociationManager::s_lastError;

bool FileAssociationManager::isRunningAsAdmin()
{
    BOOL isAdmin = FALSE;
    PSID adminGroup = NULL;
    SID_IDENTIFIER_AUTHORITY ntAuthority = SECURITY_NT_AUTHORITY;
    
    if (AllocateAndInitializeSid(&ntAuthority, 2,
                                  SECURITY_BUILTIN_DOMAIN_RID,
                                  DOMAIN_ALIAS_RID_ADMINS,
                                  0, 0, 0, 0, 0, 0,
                                  &adminGroup))
    {
        CheckTokenMembership(NULL, adminGroup, &isAdmin);
        FreeSid(adminGroup);
    }
    
    return isAdmin == TRUE;
}

QString FileAssociationManager::getLastError()
{
    return s_lastError;
}

bool FileAssociationManager::setRegistryValue(const QString &keyPath, const QString &valueName, const QString &valueData)
{
    HKEY hKey;
    LONG result;
    
    // Parse root key
    QString rootKeyName = keyPath.split('\\').first();
    HKEY rootKey = HKEY_CLASSES_ROOT;
    
    if (rootKeyName == "HKEY_CLASSES_ROOT" || rootKeyName == "HKCR") {
        rootKey = HKEY_CLASSES_ROOT;
    } else if (rootKeyName == "HKEY_CURRENT_USER" || rootKeyName == "HKCU") {
        rootKey = HKEY_CURRENT_USER;
    } else if (rootKeyName == "HKEY_LOCAL_MACHINE" || rootKeyName == "HKLM") {
        rootKey = HKEY_LOCAL_MACHINE;
    }
    
    // Get subkey path (everything after root)
    QString subKeyPath = keyPath;
    int firstBackslash = subKeyPath.indexOf('\\');
    if (firstBackslash != -1) {
        subKeyPath = subKeyPath.mid(firstBackslash + 1);
    }
    
    // Create or open the key
    result = RegCreateKeyExW(
        rootKey,
        reinterpret_cast<const wchar_t*>(subKeyPath.utf16()),
        0,
        NULL,
        REG_OPTION_NON_VOLATILE,
        KEY_WRITE,
        NULL,
        &hKey,
        NULL
    );
    
    if (result != ERROR_SUCCESS) {
        s_lastError = QString("Failed to create registry key: %1 (Error: %2)")
                          .arg(keyPath)
                          .arg(result);
        return false;
    }
    
    // Set the value
    const wchar_t* valueNameW = valueName.isEmpty() ? NULL : reinterpret_cast<const wchar_t*>(valueName.utf16());
    const wchar_t* valueDataW = reinterpret_cast<const wchar_t*>(valueData.utf16());
    DWORD dataSize = (valueData.length() + 1) * sizeof(wchar_t);
    
    result = RegSetValueExW(
        hKey,
        valueNameW,
        0,
        REG_SZ,
        reinterpret_cast<const BYTE*>(valueDataW),
        dataSize
    );
    
    RegCloseKey(hKey);
    
    if (result != ERROR_SUCCESS) {
        s_lastError = QString("Failed to set registry value: %1\\%2 (Error: %3)")
                          .arg(keyPath)
                          .arg(valueName)
                          .arg(result);
        return false;
    }
    
    return true;
}

QString FileAssociationManager::getRegistryValue(const QString &keyPath, const QString &valueName)
{
    HKEY hKey;
    LONG result;
    
    // Parse root key
    QString rootKeyName = keyPath.split('\\').first();
    HKEY rootKey = HKEY_CLASSES_ROOT;
    
    if (rootKeyName == "HKEY_CLASSES_ROOT" || rootKeyName == "HKCR") {
        rootKey = HKEY_CLASSES_ROOT;
    } else if (rootKeyName == "HKEY_CURRENT_USER" || rootKeyName == "HKCU") {
        rootKey = HKEY_CURRENT_USER;
    } else if (rootKeyName == "HKEY_LOCAL_MACHINE" || rootKeyName == "HKLM") {
        rootKey = HKEY_LOCAL_MACHINE;
    }
    
    // Get subkey path
    QString subKeyPath = keyPath;
    int firstBackslash = subKeyPath.indexOf('\\');
    if (firstBackslash != -1) {
        subKeyPath = subKeyPath.mid(firstBackslash + 1);
    }
    
    // Open the key
    result = RegOpenKeyExW(
        rootKey,
        reinterpret_cast<const wchar_t*>(subKeyPath.utf16()),
        0,
        KEY_READ,
        &hKey
    );
    
    if (result != ERROR_SUCCESS) {
        return QString();
    }
    
    // Read the value
    wchar_t buffer[512];
    DWORD bufferSize = sizeof(buffer);
    DWORD type;
    
    const wchar_t* valueNameW = valueName.isEmpty() ? NULL : reinterpret_cast<const wchar_t*>(valueName.utf16());
    
    result = RegQueryValueExW(
        hKey,
        valueNameW,
        NULL,
        &type,
        reinterpret_cast<BYTE*>(buffer),
        &bufferSize
    );
    
    RegCloseKey(hKey);
    
    if (result != ERROR_SUCCESS || type != REG_SZ) {
        return QString();
    }
    
    return QString::fromWCharArray(buffer);
}

bool FileAssociationManager::registryKeyExists(const QString &keyPath)
{
    HKEY hKey;
    LONG result;
    
    // Parse root key
    QString rootKeyName = keyPath.split('\\').first();
    HKEY rootKey = HKEY_CLASSES_ROOT;
    
    if (rootKeyName == "HKEY_CLASSES_ROOT" || rootKeyName == "HKCR") {
        rootKey = HKEY_CLASSES_ROOT;
    } else if (rootKeyName == "HKEY_CURRENT_USER" || rootKeyName == "HKCU") {
        rootKey = HKEY_CURRENT_USER;
    } else if (rootKeyName == "HKEY_LOCAL_MACHINE" || rootKeyName == "HKLM") {
        rootKey = HKEY_LOCAL_MACHINE;
    }
    
    // Get subkey path
    QString subKeyPath = keyPath;
    int firstBackslash = subKeyPath.indexOf('\\');
    if (firstBackslash != -1) {
        subKeyPath = subKeyPath.mid(firstBackslash + 1);
    }
    
    // Try to open the key
    result = RegOpenKeyExW(
        rootKey,
        reinterpret_cast<const wchar_t*>(subKeyPath.utf16()),
        0,
        KEY_READ,
        &hKey
    );
    
    if (result == ERROR_SUCCESS) {
        RegCloseKey(hKey);
        return true;
    }
    
    return false;
}

bool FileAssociationManager::deleteRegistryKey(const QString &keyPath)
{
    LONG result;
    
    // Parse root key
    QString rootKeyName = keyPath.split('\\').first();
    HKEY rootKey = HKEY_CLASSES_ROOT;
    
    if (rootKeyName == "HKEY_CLASSES_ROOT" || rootKeyName == "HKCR") {
        rootKey = HKEY_CLASSES_ROOT;
    } else if (rootKeyName == "HKEY_CURRENT_USER" || rootKeyName == "HKCU") {
        rootKey = HKEY_CURRENT_USER;
    } else if (rootKeyName == "HKEY_LOCAL_MACHINE" || rootKeyName == "HKLM") {
        rootKey = HKEY_LOCAL_MACHINE;
    }
    
    // Get subkey path
    QString subKeyPath = keyPath;
    int firstBackslash = subKeyPath.indexOf('\\');
    if (firstBackslash != -1) {
        subKeyPath = subKeyPath.mid(firstBackslash + 1);
    }
    
    // Delete the key and all subkeys recursively
    result = RegDeleteTreeW(
        rootKey,
        reinterpret_cast<const wchar_t*>(subKeyPath.utf16())
    );
    
    if (result != ERROR_SUCCESS && result != ERROR_FILE_NOT_FOUND) {
        s_lastError = QString("Failed to delete registry key: %1 (Error: %2)")
                          .arg(keyPath)
                          .arg(result);
        return false;
    }
    
    return true;
}

QList<FileTypeInfo> FileAssociationManager::buildFileTypeList()
{
    QList<FileTypeInfo> fileTypes;
    
    // LZ4
    fileTypes.append({
        ".lz4",
        "nvCOMP.LZ4Archive",
        "LZ4 Compressed Archive",
        "1",  // Icon index
        "LZ4"
    });
    
    // Zstd
    fileTypes.append({
        ".zstd",
        "nvCOMP.ZstdArchive",
        "Zstandard Compressed Archive",
        "2",
        "Zstd"
    });
    
    // Snappy
    fileTypes.append({
        ".snappy",
        "nvCOMP.SnappyArchive",
        "Snappy Compressed Archive",
        "3",
        "Snappy"
    });
    
    // Generic nvCOMP
    fileTypes.append({
        ".nvcomp",
        "nvCOMP.Archive",
        "nvCOMP Compressed Archive",
        "0",  // Main icon
        "Auto"
    });
    
    // GPU algorithms
    fileTypes.append({
        ".gdeflate",
        "nvCOMP.GDeflateArchive",
        "GDeflate GPU Compressed Archive",
        "4",
        "GDeflate"
    });
    
    fileTypes.append({
        ".ans",
        "nvCOMP.ANSArchive",
        "ANS GPU Compressed Archive",
        "5",
        "ANS"
    });
    
    fileTypes.append({
        ".bitcomp",
        "nvCOMP.BitcompArchive",
        "Bitcomp GPU Compressed Archive",
        "6",
        "Bitcomp"
    });
    
    return fileTypes;
}

QList<FileTypeInfo> FileAssociationManager::getSupportedFileTypes()
{
    return buildFileTypeList();
}

bool FileAssociationManager::createProgId(
    const QString &progId,
    const QString &description,
    const QString &exePath,
    int iconIndex)
{
    // Create ProgID key with description
    QString progIdPath = QString("HKEY_CLASSES_ROOT\\%1").arg(progId);
    if (!setRegistryValue(progIdPath, "", description)) {
        return false;
    }
    
    // Set default icon
    QString iconPath = QString("HKEY_CLASSES_ROOT\\%1\\DefaultIcon").arg(progId);
    QString iconValue = QString("\"%1\",%2").arg(exePath).arg(iconIndex);
    if (!setRegistryValue(iconPath, "", iconValue)) {
        return false;
    }
    
    return true;
}

bool FileAssociationManager::addShellCommands(const QString &progId, const QString &exePath)
{
    // Normalize path to Windows format (backslashes)
    QString windowsPath = QDir::toNativeSeparators(exePath);
    
    // Create "nvCOMP" cascading menu (like WinRAR style)
    QString nvcompMenuPath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\nvCOMP").arg(progId);
    
    // Set the display name using MUIVerb (required for cascading menus)
    if (!setRegistryValue(nvcompMenuPath, "MUIVerb", "nvCOMP")) {
        return false;
    }
    
    // Set icon for the main menu
    if (!setRegistryValue(nvcompMenuPath, "Icon", QString("\"%1\",0").arg(windowsPath))) {
        return false;
    }
    
    // Enable cascading menu (this tells Windows to look for submenus)
    if (!setRegistryValue(nvcompMenuPath, "SubCommands", "")) {
        return false;
    }
    
    // Add "View Archive" as first option in submenu
    QString viewArchivePath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\nvCOMP\\shell\\view").arg(progId);
    if (!setRegistryValue(viewArchivePath, "MUIVerb", "&View Archive")) {
        return false;
    }
    
    QString viewArchiveCommandPath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\nvCOMP\\shell\\view\\command").arg(progId);
    QString viewArchiveCommand = QString("\"%1\" \"%2\"").arg(windowsPath, "%1");
    if (!setRegistryValue(viewArchiveCommandPath, "", viewArchiveCommand)) {
        return false;
    }
    
    // Add "Extract Here" command under nvCOMP menu
    QString extractHerePath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\nvCOMP\\shell\\extract_here").arg(progId);
    if (!setRegistryValue(extractHerePath, "MUIVerb", "Extract &Here")) {
        return false;
    }
    
    QString extractHereCommandPath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\nvCOMP\\shell\\extract_here\\command").arg(progId);
    QString extractHereCommand = QString("\"%1\" --extract-here \"%2\"").arg(windowsPath, "%1");
    if (!setRegistryValue(extractHereCommandPath, "", extractHereCommand)) {
        return false;
    }
    
    // Add "Extract to Folder" command under nvCOMP menu
    QString extractToPath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\nvCOMP\\shell\\extract_to").arg(progId);
    if (!setRegistryValue(extractToPath, "MUIVerb", "Extract to &Folder...")) {
        return false;
    }
    
    QString extractToCommandPath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\nvCOMP\\shell\\extract_to\\command").arg(progId);
    QString extractToCommand = QString("\"%1\" --decompress \"%2\"").arg(windowsPath, "%1");
    if (!setRegistryValue(extractToCommandPath, "", extractToCommand)) {
        return false;
    }
    
    // Set "open" as the default double-click action (must be outside the submenu)
    QString openCommandPath = QString("HKEY_CLASSES_ROOT\\%1\\shell\\open\\command").arg(progId);
    QString openCommand = QString("\"%1\" \"%2\"").arg(windowsPath, "%1");
    if (!setRegistryValue(openCommandPath, "", openCommand)) {
        return false;
    }
    
    return true;
}

bool FileAssociationManager::associateExtensionWithProgId(const QString &extension, const QString &progId)
{
    QString extPath = QString("HKEY_CLASSES_ROOT\\%1").arg(extension);
    return setRegistryValue(extPath, "", progId);
}

bool FileAssociationManager::registerAssociation(
    const QString &extension,
    const QString &progId,
    const QString &description,
    const QString &exePath,
    int iconIndex)
{
    s_lastError.clear();
    
    if (!isRunningAsAdmin()) {
        s_lastError = "Administrator privileges required for file association registration.";
        return false;
    }
    
    // Create ProgID with icon
    if (!createProgId(progId, description, exePath, iconIndex)) {
        return false;
    }
    
    // Add shell commands
    if (!addShellCommands(progId, exePath)) {
        return false;
    }
    
    // Associate extension with ProgID
    if (!associateExtensionWithProgId(extension, progId)) {
        return false;
    }
    
    // Notify shell of changes
    notifyShellAssociationChanged();
    
    qDebug() << "Registered file association:" << extension << "->" << progId;
    return true;
}

bool FileAssociationManager::registerAllAssociations(const QString &exePath)
{
    s_lastError.clear();
    
    if (!isRunningAsAdmin()) {
        s_lastError = "Administrator privileges required for file association registration.";
        return false;
    }
    
    QList<FileTypeInfo> fileTypes = getSupportedFileTypes();
    bool allSuccess = true;
    QStringList failedExtensions;
    
    for (const FileTypeInfo &fileType : fileTypes) {
        bool success = registerAssociation(
            fileType.extension,
            fileType.progId,
            fileType.description,
            exePath,
            fileType.iconIndex.toInt()
        );
        
        if (!success) {
            allSuccess = false;
            failedExtensions.append(fileType.extension);
            qWarning() << "Failed to register association for" << fileType.extension;
        }
    }
    
    if (!allSuccess) {
        s_lastError = QString("Failed to register some associations: %1")
                          .arg(failedExtensions.join(", "));
    }
    
    return allSuccess;
}

bool FileAssociationManager::unregisterAssociation(const QString &extension)
{
    s_lastError.clear();
    
    if (!isRunningAsAdmin()) {
        s_lastError = "Administrator privileges required for file association removal.";
        return false;
    }
    
    // Get the current ProgID for this extension
    QString extPath = QString("HKEY_CLASSES_ROOT\\%1").arg(extension);
    QString currentProgId = getRegistryValue(extPath, "");
    
    // Only remove if it's an nvCOMP ProgID
    if (!currentProgId.startsWith("nvCOMP.")) {
        qDebug() << "Extension" << extension << "is not associated with nvCOMP, skipping";
        return true;
    }
    
    // Delete the ProgID
    QString progIdPath = QString("HKEY_CLASSES_ROOT\\%1").arg(currentProgId);
    if (!deleteRegistryKey(progIdPath)) {
        return false;
    }
    
    // Delete the extension association
    if (!deleteRegistryKey(extPath)) {
        return false;
    }
    
    // Notify shell of changes
    notifyShellAssociationChanged();
    
    qDebug() << "Unregistered file association:" << extension;
    return true;
}

bool FileAssociationManager::unregisterAllAssociations()
{
    s_lastError.clear();
    
    if (!isRunningAsAdmin()) {
        s_lastError = "Administrator privileges required for file association removal.";
        return false;
    }
    
    QList<FileTypeInfo> fileTypes = getSupportedFileTypes();
    bool allSuccess = true;
    QStringList failedExtensions;
    
    for (const FileTypeInfo &fileType : fileTypes) {
        bool success = unregisterAssociation(fileType.extension);
        
        if (!success) {
            allSuccess = false;
            failedExtensions.append(fileType.extension);
            qWarning() << "Failed to unregister association for" << fileType.extension;
        }
    }
    
    if (!allSuccess) {
        s_lastError = QString("Failed to unregister some associations: %1")
                          .arg(failedExtensions.join(", "));
    }
    
    return allSuccess;
}

bool FileAssociationManager::isAssociated(const QString &extension)
{
    QString extPath = QString("HKEY_CLASSES_ROOT\\%1").arg(extension);
    QString progId = getRegistryValue(extPath, "");
    
    return progId.startsWith("nvCOMP.");
}

bool FileAssociationManager::areAllAssociated()
{
    QList<FileTypeInfo> fileTypes = getSupportedFileTypes();
    
    for (const FileTypeInfo &fileType : fileTypes) {
        if (!isAssociated(fileType.extension)) {
            return false;
        }
    }
    
    return true;
}

bool FileAssociationManager::setAsDefault(const QString &extension)
{
    s_lastError.clear();
    
    if (!isRunningAsAdmin()) {
        s_lastError = "Administrator privileges required to set default program.";
        return false;
    }
    
    // Find the ProgID for this extension
    QList<FileTypeInfo> fileTypes = getSupportedFileTypes();
    QString progId;
    
    for (const FileTypeInfo &fileType : fileTypes) {
        if (fileType.extension == extension) {
            progId = fileType.progId;
            break;
        }
    }
    
    if (progId.isEmpty()) {
        s_lastError = QString("Extension %1 is not supported by nvCOMP").arg(extension);
        return false;
    }
    
    // Set as default
    QString extPath = QString("HKEY_CLASSES_ROOT\\%1").arg(extension);
    if (!setRegistryValue(extPath, "", progId)) {
        return false;
    }
    
    // Notify shell of changes
    notifyShellAssociationChanged();
    
    return true;
}

QString FileAssociationManager::getAssociatedProgId(const QString &extension)
{
    QString extPath = QString("HKEY_CLASSES_ROOT\\%1").arg(extension);
    return getRegistryValue(extPath, "");
}

void FileAssociationManager::notifyShellAssociationChanged()
{
    // Notify Windows that file associations have changed
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, NULL, NULL);
}

#else // Not Windows

// Stubs for non-Windows platforms

QString FileAssociationManager::s_lastError = "File associations are only supported on Windows";

bool FileAssociationManager::isRunningAsAdmin() { return false; }
QString FileAssociationManager::getLastError() { return s_lastError; }
bool FileAssociationManager::registerAssociation(const QString &, const QString &, const QString &, const QString &, int) { return false; }
bool FileAssociationManager::registerAllAssociations(const QString &) { return false; }
bool FileAssociationManager::unregisterAssociation(const QString &) { return false; }
bool FileAssociationManager::unregisterAllAssociations() { return false; }
bool FileAssociationManager::isAssociated(const QString &) { return false; }
bool FileAssociationManager::areAllAssociated() { return false; }
bool FileAssociationManager::setAsDefault(const QString &) { return false; }
QString FileAssociationManager::getAssociatedProgId(const QString &) { return QString(); }
void FileAssociationManager::notifyShellAssociationChanged() {}

QList<FileTypeInfo> FileAssociationManager::getSupportedFileTypes()
{
    return QList<FileTypeInfo>();
}

bool FileAssociationManager::setRegistryValue(const QString &, const QString &, const QString &) { return false; }
QString FileAssociationManager::getRegistryValue(const QString &, const QString &) { return QString(); }
bool FileAssociationManager::registryKeyExists(const QString &) { return false; }
bool FileAssociationManager::deleteRegistryKey(const QString &) { return false; }
bool FileAssociationManager::createProgId(const QString &, const QString &, const QString &, int) { return false; }
bool FileAssociationManager::addShellCommands(const QString &, const QString &) { return false; }
bool FileAssociationManager::associateExtensionWithProgId(const QString &, const QString &) { return false; }
QList<FileTypeInfo> FileAssociationManager::buildFileTypeList() { return QList<FileTypeInfo>(); }

#endif // _WIN32

