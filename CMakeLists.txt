cmake_minimum_required(VERSION 3.18)
project(NvCompCLI LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Build Options
# ============================================================================

option(BUILD_CLI "Build the command-line interface" ON)
option(BUILD_GUI "Build the Qt GUI application" OFF)
option(BUILD_TESTS "Build test executables" ON)

message(STATUS "Build configuration:")
message(STATUS "  BUILD_CLI:   ${BUILD_CLI}")
message(STATUS "  BUILD_GUI:   ${BUILD_GUI}")
message(STATUS "  BUILD_TESTS: ${BUILD_TESTS}")

include(FetchContent)

# ============================================================================
# Fetch nvCOMP SDK
# ============================================================================

message(STATUS "Detecting platform for nvCOMP download...")

if(WIN32)
    set(NVCOMP_URL "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/windows-x86_64/nvcomp-windows-x86_64-5.1.0.21_cuda13-archive.zip")
    message(STATUS "Platform: Windows")
elseif(UNIX AND NOT APPLE)
    set(NVCOMP_URL "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda13-archive.tar.xz")
    message(STATUS "Platform: Linux")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

FetchContent_Declare(
    nvcomp_archive
    URL ${NVCOMP_URL}
)

FetchContent_GetProperties(nvcomp_archive)
if(NOT nvcomp_archive_POPULATED)
    message(STATUS "Downloading nvCOMP SDK...")
    FetchContent_Populate(nvcomp_archive)
endif()

# Find nvCOMP path
if(EXISTS "${nvcomp_archive_SOURCE_DIR}/include/nvcomp.hpp")
    set(NVCOMP_PATH "${nvcomp_archive_SOURCE_DIR}")
else()
    file(GLOB CHILDREN "${nvcomp_archive_SOURCE_DIR}/*")
    foreach(CHILD ${CHILDREN})
        if(IS_DIRECTORY "${CHILD}" AND EXISTS "${CHILD}/include/nvcomp.hpp")
            set(NVCOMP_PATH "${CHILD}")
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED NVCOMP_PATH)
    message(FATAL_ERROR "Could not find nvCOMP in ${nvcomp_archive_SOURCE_DIR}")
endif()

message(STATUS "nvCOMP path: ${NVCOMP_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${NVCOMP_PATH}/lib/cmake/nvcomp")

# ============================================================================
# Fetch and Patch LZ4
# ============================================================================

message(STATUS "Fetching LZ4...")

FetchContent_Declare(
    lz4
    URL https://github.com/lz4/lz4/archive/refs/tags/v1.9.4.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(lz4)
if(NOT lz4_POPULATED)
    FetchContent_Populate(lz4)
    
    # Patch CMakeLists.txt to use compatible CMake version
    set(LZ4_CMAKE_FILE "${lz4_SOURCE_DIR}/build/cmake/CMakeLists.txt")
    if(EXISTS ${LZ4_CMAKE_FILE})
        file(READ ${LZ4_CMAKE_FILE} LZ4_CMAKE_CONTENT)
        string(REGEX REPLACE "cmake_minimum_required\\(VERSION [^)]+\\)" 
               "cmake_minimum_required(VERSION 3.5)" 
               LZ4_CMAKE_CONTENT "${LZ4_CMAKE_CONTENT}")
        file(WRITE ${LZ4_CMAKE_FILE} "${LZ4_CMAKE_CONTENT}")
        message(STATUS "Patched LZ4 CMakeLists.txt")
    endif()
    
    # Build LZ4
    set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
    set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
    add_subdirectory(${lz4_SOURCE_DIR}/build/cmake ${lz4_BINARY_DIR})
endif()

# ============================================================================
# Fetch and Patch Snappy
# ============================================================================

message(STATUS "Fetching Snappy...")

FetchContent_Declare(
    snappy
    URL https://github.com/google/snappy/archive/refs/tags/1.2.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(snappy)
if(NOT snappy_POPULATED)
    FetchContent_Populate(snappy)
    
    # Patch CMakeLists.txt
    set(SNAPPY_CMAKE_FILE "${snappy_SOURCE_DIR}/CMakeLists.txt")
    if(EXISTS ${SNAPPY_CMAKE_FILE})
        file(READ ${SNAPPY_CMAKE_FILE} SNAPPY_CMAKE_CONTENT)
        string(REGEX REPLACE "cmake_minimum_required\\(VERSION [^)]+\\)" 
               "cmake_minimum_required(VERSION 3.5)" 
               SNAPPY_CMAKE_CONTENT "${SNAPPY_CMAKE_CONTENT}")
        file(WRITE ${SNAPPY_CMAKE_FILE} "${SNAPPY_CMAKE_CONTENT}")
        message(STATUS "Patched Snappy CMakeLists.txt")
    endif()
    
    # Build Snappy
    set(SNAPPY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SNAPPY_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${snappy_SOURCE_DIR} ${snappy_BINARY_DIR})
endif()

# ============================================================================
# Fetch and Patch Zstd
# ============================================================================

message(STATUS "Fetching Zstd...")

FetchContent_Declare(
    zstd
    URL https://github.com/facebook/zstd/archive/refs/tags/v1.5.5.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(zstd)
if(NOT zstd_POPULATED)
    FetchContent_Populate(zstd)
    
    # Patch CMakeLists.txt
    set(ZSTD_CMAKE_FILE "${zstd_SOURCE_DIR}/build/cmake/CMakeLists.txt")
    if(EXISTS ${ZSTD_CMAKE_FILE})
        file(READ ${ZSTD_CMAKE_FILE} ZSTD_CMAKE_CONTENT)
        string(REGEX REPLACE "cmake_minimum_required\\(VERSION [^)]+\\)" 
               "cmake_minimum_required(VERSION 3.5)" 
               ZSTD_CMAKE_CONTENT "${ZSTD_CMAKE_CONTENT}")
        file(WRITE ${ZSTD_CMAKE_FILE} "${ZSTD_CMAKE_CONTENT}")
        message(STATUS "Patched Zstd CMakeLists.txt")
    endif()
    
    # Build Zstd
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
    add_subdirectory(${zstd_SOURCE_DIR}/build/cmake ${zstd_BINARY_DIR})
endif()

# ============================================================================
# Find nvCOMP package
# ============================================================================

find_package(nvcomp REQUIRED)

# ============================================================================
# Find Qt6 (if building GUI)
# ============================================================================

if(BUILD_GUI)
    message(STATUS "Searching for Qt6...")
    
    # Try to find Qt6 first (user may have it installed)
    find_package(Qt6 COMPONENTS Core Widgets Gui Test QUIET)
    
    if(Qt6_FOUND)
        message(STATUS "Qt6 found: ${Qt6_VERSION}")
        message(STATUS "  Qt6Core:    ${Qt6Core_DIR}")
        message(STATUS "  Qt6Widgets: ${Qt6Widgets_DIR}")
        message(STATUS "  Qt6Gui:     ${Qt6Gui_DIR}")
        message(STATUS "  Qt6Test:    ${Qt6Test_DIR}")
    else()
        message(STATUS "Qt6 not found. Attempting automatic installation...")
        
        # Check if Python is available
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(NOT Python3_FOUND)
            message(FATAL_ERROR 
                "Qt6 not found and Python3 is required for automatic installation.\n"
                "\n"
                "Please either:\n"
                "1. Install Python 3.6+: https://www.python.org/downloads/\n"
                "2. Install Qt6 manually: https://www.qt.io/download-open-source\n"
                "3. Skip GUI build: cmake -B build -DBUILD_GUI=OFF\n"
                "\n"
                "See gui/INSTALL_QT6.md for detailed instructions.")
        endif()
        
        # Install aqtinstall if not present
        message(STATUS "Checking for aqtinstall...")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pip show aqtinstall
            RESULT_VARIABLE AQTINSTALL_CHECK
            OUTPUT_QUIET
            ERROR_QUIET
        )
        
        if(NOT AQTINSTALL_CHECK EQUAL 0)
            message(STATUS "Installing aqtinstall (Qt installer tool)...")
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -m pip install -q aqtinstall
                RESULT_VARIABLE AQTINSTALL_INSTALL_RESULT
                ERROR_QUIET
            )
            if(NOT AQTINSTALL_INSTALL_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to install aqtinstall. Please install Qt6 manually.")
            endif()
            message(STATUS "aqtinstall installed successfully")
        endif()
        
        # Determine Qt installation directory
        set(QT_INSTALL_DIR "${CMAKE_BINARY_DIR}/qt6")
        
        # Determine platform and architecture
        if(WIN32)
            if(MSVC)
                if(MSVC_VERSION GREATER_EQUAL 1930)
                    # VS 2022 - try both 2022 and 2019 architectures
                    set(QT_ARCH_LIST "win64_msvc2022_64" "win64_msvc2019_64")
                    set(QT_VERSION_LIST "6.8.0" "6.7.3" "6.6.3")
                elseif(MSVC_VERSION GREATER_EQUAL 1920)
                    # VS 2019
                    set(QT_ARCH_LIST "win64_msvc2019_64")
                    set(QT_VERSION_LIST "6.7.3" "6.6.3" "6.5.3")
                else()
                    message(FATAL_ERROR "Unsupported MSVC version. Please use VS 2019 or later.")
                endif()
            else()
                message(FATAL_ERROR "Only MSVC compiler is supported for Windows Qt builds.")
            endif()
            set(QT_OS "windows")
        elseif(UNIX AND NOT APPLE)
            set(QT_OS "linux")
            set(QT_ARCH_LIST "gcc_64")
            set(QT_VERSION_LIST "6.7.3" "6.6.3" "6.5.3")
        else()
            message(FATAL_ERROR "Unsupported platform for automatic Qt installation.")
        endif()
        
        # Try to install Qt (try multiple versions and architectures)
        set(QT_INSTALLED FALSE)
        
        foreach(QT_VERSION ${QT_VERSION_LIST})
            if(QT_INSTALLED)
                break()
            endif()
            
            foreach(QT_ARCH ${QT_ARCH_LIST})
                if(QT_INSTALLED)
                    break()
                endif()
                
                message(STATUS "Attempting to download Qt ${QT_VERSION} (${QT_ARCH})...")
                message(STATUS "  This may take 5-10 minutes (~1-2 GB download)...")
                message(STATUS "  Installation directory: ${QT_INSTALL_DIR}")
                
                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -m aqt install-qt 
                        ${QT_OS} desktop ${QT_VERSION} ${QT_ARCH}
                        -O ${QT_INSTALL_DIR}
                    RESULT_VARIABLE QT_INSTALL_RESULT
                    OUTPUT_VARIABLE QT_INSTALL_OUTPUT
                    ERROR_VARIABLE QT_INSTALL_ERROR
                    TIMEOUT 1800  # 30 minutes timeout
                )
                
                if(QT_INSTALL_RESULT EQUAL 0)
                    message(STATUS "Qt ${QT_VERSION} (${QT_ARCH}) installed successfully!")
                    
                    # Extract the directory name (remove win64_ prefix if present)
                    string(REPLACE "win64_" "" QT_ARCH_DIR "${QT_ARCH}")
                    
                    # Set Qt6 path for find_package
                    set(Qt6_DIR "${QT_INSTALL_DIR}/${QT_VERSION}/${QT_ARCH_DIR}/lib/cmake/Qt6")
                    list(APPEND CMAKE_PREFIX_PATH "${QT_INSTALL_DIR}/${QT_VERSION}/${QT_ARCH_DIR}")
                    
                    message(STATUS "Qt6_DIR set to: ${Qt6_DIR}")
                    set(QT_INSTALLED TRUE)
                    set(INSTALLED_QT_VERSION ${QT_VERSION})
                    set(INSTALLED_QT_ARCH ${QT_ARCH})
                    break()
                else()
                    message(STATUS "  Failed to install Qt ${QT_VERSION} (${QT_ARCH})")
                    message(STATUS "  Error: ${QT_INSTALL_ERROR}")
                    message(STATUS "  Trying next option...")
                endif()
            endforeach()
        endforeach()
        
        if(NOT QT_INSTALLED)
            message(FATAL_ERROR 
                "Failed to install Qt6 automatically after trying multiple versions.\n"
                "\n"
                "Manual installation options:\n"
                "1. Qt Online Installer: https://www.qt.io/download-open-source\n"
                "2. Manual aqtinstall command:\n"
                "   python -m aqt install-qt windows desktop 6.8.0 win64_msvc2022_64 -O C:/Qt\n"
                "   Then: cmake -B build_gui -DBUILD_GUI=ON -DCMAKE_PREFIX_PATH=\"C:/Qt/6.8.0/msvc2022_64\"\n"
                "3. See gui/INSTALL_QT6.md for detailed instructions\n"
                "\n"
                "Or skip GUI build: cmake -B build -DBUILD_GUI=OFF")
        endif()
        
        # Now try to find Qt6 again
        find_package(Qt6 COMPONENTS Core Widgets Gui Test REQUIRED)
        message(STATUS "Qt6 ${INSTALLED_QT_VERSION} successfully configured!")
    endif()
endif()

# ============================================================================
# Build Core Library
# ============================================================================

# Pass dependency include directories to core library via parent scope includes
# This allows core to find lz4, snappy, zstd headers
include_directories(
    ${lz4_SOURCE_DIR}/lib
    ${snappy_SOURCE_DIR}
    ${snappy_BINARY_DIR}
    ${zstd_SOURCE_DIR}/lib
)

# Build core library subdirectory (always built - required by CLI/GUI/tests)
add_subdirectory(core)

# ============================================================================
# Create CLI executable
# ============================================================================

if(BUILD_CLI)
    message(STATUS "Configuring CLI executable...")
    
    add_executable(nvcomp_cli main.cu)

    # Set CUDA architectures
    set_property(TARGET nvcomp_cli PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)

    # Include directories
    # The CLI only needs the core library C API headers
    target_include_directories(nvcomp_cli PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    )

    # Link libraries
    # The CLI is now a thin wrapper around the core library (Task 1.3)
    # All compression logic is handled by nvcomp_core
    target_link_libraries(nvcomp_cli PRIVATE 
        nvcomp_core
    )

    # ========================================================================
    # Platform-specific configuration
    # ========================================================================

    if(WIN32)
        # Windows: Copy DLLs to output directory
        add_custom_command(TARGET nvcomp_cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVCOMP_PATH}/bin/nvcomp64_5.dll"
            $<TARGET_FILE_DIR:nvcomp_cli>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVCOMP_PATH}/bin/nvcomp_cpu64_5.dll"
            $<TARGET_FILE_DIR:nvcomp_cli>
            COMMENT "Copying nvCOMP DLLs to CLI output directory"
        )
        
        # Copy core library DLL to CLI directory
        add_custom_command(TARGET nvcomp_cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nvcomp_core>
            $<TARGET_FILE_DIR:nvcomp_cli>
            COMMENT "Copying nvcomp_core DLL to CLI output directory"
        )
    else()
        # Linux: Set RPATH to find shared libraries
        set_target_properties(nvcomp_cli PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
            BUILD_WITH_INSTALL_RPATH FALSE
            LINK_FLAGS "-Wl,--disable-new-dtags"
        )
    endif()
    
    # ========================================================================
    # Installation
    # ========================================================================
    
    install(TARGETS nvcomp_cli
        RUNTIME DESTINATION bin
        COMPONENT cli
    )
    
    message(STATUS "CLI executable configured successfully")
endif()

# ============================================================================
# Create Test Executables
# ============================================================================

if(BUILD_TESTS)
    message(STATUS "Configuring test executables...")
    
    enable_testing()
    
    add_executable(test_c_api unit_test/test_c_api.cpp)

    # Include directories
    target_include_directories(test_c_api PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    )

    # Link libraries
    target_link_libraries(test_c_api PRIVATE 
        nvcomp_core
    )

    # ========================================================================
    # Platform-specific configuration for tests
    # ========================================================================

    if(WIN32)
        # Windows: Copy DLLs to test output directory
        add_custom_command(TARGET test_c_api POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nvcomp_core>
            $<TARGET_FILE_DIR:test_c_api>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVCOMP_PATH}/bin/nvcomp64_5.dll"
            $<TARGET_FILE_DIR:test_c_api>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVCOMP_PATH}/bin/nvcomp_cpu64_5.dll"
            $<TARGET_FILE_DIR:test_c_api>
            COMMENT "Copying DLLs to test output directory"
        )
    else()
        # Linux: Set RPATH for tests
        set_target_properties(test_c_api PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
            BUILD_WITH_INSTALL_RPATH FALSE
        )
    endif()
    
    # ========================================================================
    # Register tests with CTest
    # ========================================================================
    
    add_test(NAME test_c_api COMMAND test_c_api)
    
    message(STATUS "Test executables configured successfully")
endif()

# ============================================================================
# GUI Application (Future - Phase 2)
# ============================================================================

if(BUILD_GUI)
    message(STATUS "Configuring GUI application...")
    
    # GUI will be added in Phase 2 (Task 2.1+)
    # For now, just check that Qt6 is available
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gui/CMakeLists.txt")
        add_subdirectory(gui)
        message(STATUS "GUI application configured successfully")
    else()
        message(WARNING "BUILD_GUI is ON but gui/ directory not found. GUI will be added in Phase 2.")
    endif()
endif()

# ============================================================================
# Installation Configuration
# ============================================================================

# Install core library (always)
# Core library installation is handled in core/CMakeLists.txt

# Install nvCOMP SDK DLLs/shared libraries on Windows
if(WIN32)
    install(FILES
        "${NVCOMP_PATH}/bin/nvcomp64_5.dll"
        "${NVCOMP_PATH}/bin/nvcomp_cpu64_5.dll"
        DESTINATION bin
        COMPONENT runtime
    )
endif()

# Linux: nvCOMP .so files should be in a known location or use RPATH
# The RPATH is configured per-target above

# Set installation directories
include(GNUInstallDirs)

# Installation summary
message(STATUS "Installation configuration:")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Binaries:       ${CMAKE_INSTALL_BINDIR}")
message(STATUS "  Libraries:      ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  Headers:        ${CMAKE_INSTALL_INCLUDEDIR}")

message(STATUS "")
message(STATUS "========================================")
message(STATUS "nvCOMP Project Configuration Complete")
message(STATUS "========================================")
message(STATUS "Build configuration:")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  CLI:           ${BUILD_CLI}")
message(STATUS "  GUI:           ${BUILD_GUI}")
message(STATUS "  Tests:         ${BUILD_TESTS}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  nvCOMP:        ${NVCOMP_PATH}")
message(STATUS "  LZ4:           ${lz4_SOURCE_DIR}")
message(STATUS "  Snappy:        ${snappy_SOURCE_DIR}")
message(STATUS "  Zstd:          ${zstd_SOURCE_DIR}")
if(BUILD_GUI)
    message(STATUS "  Qt6:           ${Qt6_VERSION}")
endif()
message(STATUS "")
message(STATUS "Targets built:")
message(STATUS "  nvcomp_core:   Always (shared library)")
if(BUILD_CLI)
    message(STATUS "  nvcomp_cli:    Enabled")
endif()
if(BUILD_GUI)
    message(STATUS "  nvcomp_gui:    Enabled (Phase 2)")
endif()
if(BUILD_TESTS)
    message(STATUS "  test_c_api:    Enabled")
endif()
message(STATUS "========================================")
message(STATUS "")
