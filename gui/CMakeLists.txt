cmake_minimum_required(VERSION 3.18)

# ============================================================================
# Qt GUI Application Configuration
# ============================================================================

project(nvcomp_gui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt specific settings
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Set AUTOUIC search paths to find .ui files in ui/ directory
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

# Ensure Qt is found (should be checked by parent CMakeLists.txt)
find_package(Qt6 COMPONENTS Core Widgets Gui Test REQUIRED)

message(STATUS "Configuring nvcomp_gui application...")
message(STATUS "  Qt6 version: ${Qt6_VERSION}")

# ============================================================================
# Source Files
# ============================================================================

set(GUI_SOURCES
    src/main.cpp
    src/mainwindow.cpp
)

set(GUI_HEADERS
    src/mainwindow.h
)

set(GUI_UI_FILES
    ui/mainwindow.ui
)

set(GUI_RESOURCES
    resources/nvcomp.qrc
)

# ============================================================================
# Create GUI Executable
# ============================================================================

add_executable(nvcomp_gui
    ${GUI_SOURCES}
    ${GUI_HEADERS}
    ${GUI_UI_FILES}
    ${GUI_RESOURCES}
)

# ============================================================================
# Include Directories
# ============================================================================

target_include_directories(nvcomp_gui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated UI headers
    ${CMAKE_SOURCE_DIR}/core/include  # Core library API
)

# ============================================================================
# Link Libraries
# ============================================================================

target_link_libraries(nvcomp_gui PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    nvcomp_core  # Link to our compression core library
)

# ============================================================================
# Platform-Specific Configuration
# ============================================================================

if(WIN32)
    # Windows: GUI application (no console window)
    set_target_properties(nvcomp_gui PROPERTIES
        WIN32_EXECUTABLE ON
    )
    
    # Copy required DLLs to GUI output directory
    add_custom_command(TARGET nvcomp_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:nvcomp_core>
        $<TARGET_FILE_DIR:nvcomp_gui>
        COMMENT "Copying nvcomp_core DLL to GUI output directory"
    )
    
    # Copy nvCOMP SDK DLLs
    get_filename_component(NVCOMP_BIN_DIR "${NVCOMP_PATH}/bin" ABSOLUTE)
    add_custom_command(TARGET nvcomp_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_BIN_DIR}/nvcomp64_5.dll"
        $<TARGET_FILE_DIR:nvcomp_gui>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_BIN_DIR}/nvcomp_cpu64_5.dll"
        $<TARGET_FILE_DIR:nvcomp_gui>
        COMMENT "Copying nvCOMP DLLs to GUI output directory"
    )
    
    # Deploy Qt dependencies automatically with windeployqt
    get_target_property(QT_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")
    
    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET nvcomp_gui POST_BUILD
            COMMAND ${WINDEPLOYQT_EXECUTABLE} --no-compiler-runtime --no-system-d3d-compiler --no-opengl-sw "$<TARGET_FILE:nvcomp_gui>"
            COMMENT "Deploying Qt dependencies with windeployqt..."
        )
        message(STATUS "  windeployqt will run automatically after build")
    else()
        message(WARNING "windeployqt not found. Qt dependencies must be deployed manually.")
        add_custom_command(TARGET nvcomp_gui POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Run manually: windeployqt $<TARGET_FILE:nvcomp_gui>"
        )
    endif()
    
elseif(UNIX AND NOT APPLE)
    # Linux: Set RPATH to find shared libraries
    set_target_properties(nvcomp_gui PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,--disable-new-dtags"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS nvcomp_gui
    RUNTIME DESTINATION bin
    COMPONENT gui
)

# ============================================================================
# GUI Tests
# ============================================================================

if(BUILD_TESTS)
    message(STATUS "Configuring GUI tests...")
    
    enable_testing()
    
    # Test executable
    add_executable(test_mainwindow
        tests/test_mainwindow.cpp
        src/mainwindow.cpp  # Include source to test
    )
    
    target_include_directories(test_mainwindow PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/core/include
    )
    
    target_link_libraries(test_mainwindow PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Gui
        Qt6::Test
        nvcomp_core
    )
    
    # Platform-specific test configuration
    if(WIN32)
        add_custom_command(TARGET test_mainwindow POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nvcomp_core>
            $<TARGET_FILE_DIR:test_mainwindow>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVCOMP_BIN_DIR}/nvcomp64_5.dll"
            $<TARGET_FILE_DIR:test_mainwindow>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVCOMP_BIN_DIR}/nvcomp_cpu64_5.dll"
            $<TARGET_FILE_DIR:test_mainwindow>
            COMMENT "Copying DLLs to test output directory"
        )
    else()
        set_target_properties(test_mainwindow PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
            BUILD_WITH_INSTALL_RPATH FALSE
        )
    endif()
    
    # Register test with CTest
    add_test(NAME test_mainwindow COMMAND test_mainwindow)
    
    message(STATUS "GUI tests configured successfully")
endif()

message(STATUS "nvcomp_gui configured successfully")

