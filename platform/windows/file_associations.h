/**
 * @file file_associations.h
 * @brief Windows file association management for nvCOMP archive types
 * 
 * Provides registry-based file association management allowing double-click
 * to open archives and adds context menu actions for extraction.
 */

#ifndef FILE_ASSOCIATIONS_H
#define FILE_ASSOCIATIONS_H

#include <QString>
#include <QStringList>
#include <QMap>

/**
 * @struct FileTypeInfo
 * @brief Information about a registered file type
 */
struct FileTypeInfo {
    QString extension;        ///< File extension (e.g., ".lz4")
    QString progId;          ///< ProgID (e.g., "nvCOMP.LZ4Archive")
    QString description;     ///< Friendly name (e.g., "LZ4 Compressed Archive")
    QString iconIndex;       ///< Icon index in executable (e.g., "1")
    QString algorithm;       ///< Compression algorithm name
};

/**
 * @class FileAssociationManager
 * @brief Manages Windows registry integration for file associations
 * 
 * Registers and unregisters file associations in the Windows registry
 * for compressed archive types. Supports custom icons per type,
 * shell actions (Open, Extract), and integration with Windows Explorer.
 */
class FileAssociationManager
{
public:
    /**
     * @brief Registers a single file association
     * @param extension File extension (with dot, e.g., ".lz4")
     * @param progId ProgID for the file type (e.g., "nvCOMP.LZ4Archive")
     * @param description Friendly name for the file type
     * @param exePath Full path to the nvcomp-gui.exe executable
     * @param iconIndex Icon index in the executable (0-based)
     * @return true if registration succeeded, false otherwise
     * 
     * Creates registry entries under:
     * - HKEY_CLASSES_ROOT\.extension
     * - HKEY_CLASSES_ROOT\ProgID
     * 
     * Requires administrator privileges on Windows.
     */
    static bool registerAssociation(
        const QString &extension,
        const QString &progId,
        const QString &description,
        const QString &exePath,
        int iconIndex = 0
    );
    
    /**
     * @brief Registers all nvCOMP file associations
     * @param exePath Full path to the nvcomp-gui.exe executable
     * @return true if all registrations succeeded, false otherwise
     * 
     * Registers the following extensions:
     * - .lz4 (LZ4 compressed archive)
     * - .zstd (Zstd compressed archive)
     * - .snappy (Snappy compressed archive)
     * - .nvcomp (generic nvCOMP archive)
     * - .gdeflate (GDeflate GPU compressed)
     * - .ans (ANS GPU compressed)
     * - .bitcomp (Bitcomp GPU compressed)
     * 
     * Each extension gets:
     * - Custom icon
     * - "Open with nvCOMP" default action
     * - "Extract here" context menu
     * - "Extract to folder" context menu
     */
    static bool registerAllAssociations(const QString &exePath);
    
    /**
     * @brief Unregisters a single file association
     * @param extension File extension to unregister (e.g., ".lz4")
     * @return true if unregistration succeeded, false otherwise
     * 
     * Removes registry entries but only if they point to nvCOMP.
     * Preserves existing associations to other programs.
     */
    static bool unregisterAssociation(const QString &extension);
    
    /**
     * @brief Unregisters all nvCOMP file associations
     * @return true if all unregistrations succeeded, false otherwise
     */
    static bool unregisterAllAssociations();
    
    /**
     * @brief Checks if an extension is associated with nvCOMP
     * @param extension File extension to check (e.g., ".lz4")
     * @return true if associated with nvCOMP, false otherwise
     */
    static bool isAssociated(const QString &extension);
    
    /**
     * @brief Checks if all nvCOMP extensions are registered
     * @return true if all are registered, false otherwise
     */
    static bool areAllAssociated();
    
    /**
     * @brief Sets nvCOMP as the default program for an extension
     * @param extension File extension (e.g., ".lz4")
     * @return true if successful, false otherwise
     * 
     * This overwrites any existing default program association.
     * Use with caution - check user preference first.
     */
    static bool setAsDefault(const QString &extension);
    
    /**
     * @brief Gets the ProgID currently associated with an extension
     * @param extension File extension (e.g., ".lz4")
     * @return ProgID string, or empty string if not associated
     */
    static QString getAssociatedProgId(const QString &extension);
    
    /**
     * @brief Gets a list of all nvCOMP file types
     * @return QList of FileTypeInfo structures
     */
    static QList<FileTypeInfo> getSupportedFileTypes();
    
    /**
     * @brief Notifies Windows Shell that file associations have changed
     * 
     * This forces Windows Explorer to refresh its file type cache.
     * Should be called after registration/unregistration.
     */
    static void notifyShellAssociationChanged();
    
    /**
     * @brief Gets the last error message from registry operations
     * @return QString containing the error description
     */
    static QString getLastError();
    
    /**
     * @brief Checks if the current process has administrator privileges
     * @return true if running as administrator, false otherwise
     * 
     * File association registration requires admin rights.
     */
    static bool isRunningAsAdmin();
    
private:
    static QString s_lastError;  ///< Stores the last error message
    
    /**
     * @brief Creates a ProgID with all necessary subkeys
     * @param progId ProgID to create (e.g., "nvCOMP.LZ4Archive")
     * @param description Friendly name
     * @param exePath Path to executable
     * @param iconIndex Icon index in executable
     * @return true if successful, false otherwise
     */
    static bool createProgId(
        const QString &progId,
        const QString &description,
        const QString &exePath,
        int iconIndex
    );
    
    /**
     * @brief Adds shell commands to a ProgID (Open, Extract, etc.)
     * @param progId ProgID to add commands to
     * @param exePath Path to executable
     * @return true if successful, false otherwise
     */
    static bool addShellCommands(const QString &progId, const QString &exePath);
    
    /**
     * @brief Associates an extension with a ProgID
     * @param extension File extension (e.g., ".lz4")
     * @param progId ProgID to associate with
     * @return true if successful, false otherwise
     */
    static bool associateExtensionWithProgId(const QString &extension, const QString &progId);
    
    /**
     * @brief Sets a registry string value
     * @param keyPath Full registry path
     * @param valueName Name of the value to set (empty for default value)
     * @param valueData String data to write
     * @return true if successful, false otherwise
     */
    static bool setRegistryValue(const QString &keyPath, const QString &valueName, const QString &valueData);
    
    /**
     * @brief Gets a registry string value
     * @param keyPath Full registry path
     * @param valueName Name of the value to read (empty for default value)
     * @return QString containing the value, or empty string on error
     */
    static QString getRegistryValue(const QString &keyPath, const QString &valueName);
    
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
    
    /**
     * @brief Builds the list of supported file types
     * @return QList of FileTypeInfo structures
     */
    static QList<FileTypeInfo> buildFileTypeList();
};

#endif // FILE_ASSOCIATIONS_H

