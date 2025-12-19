# platform/linux/CMakeLists.txt
# Linux Desktop Integration

# Only build on Linux
if(NOT UNIX OR APPLE)
    message(STATUS "Linux desktop integration is only available on Linux systems")
    return()
endif()

# Desktop integration library
add_library(linux_desktop_integration STATIC
    desktop_integration.cpp
    desktop_integration.h
)

target_link_libraries(linux_desktop_integration
    PUBLIC
        Qt6::Core
        Qt6::Gui
)

target_include_directories(linux_desktop_integration
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Installation rules for desktop integration files (optional)
# These are generated at runtime by the DesktopIntegration class

install(TARGETS linux_desktop_integration
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT gui
)


