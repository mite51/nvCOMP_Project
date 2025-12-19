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

# Install file manager integration files (Task 5.2)
# These provide right-click context menu support for Nautilus and Nemo

# Install Nautilus Python extension
install(FILES
    nautilus_extension.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/nvcomp/nautilus
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    COMPONENT gui
    OPTIONAL
)

# Install Nemo script
install(FILES
    nemo_script.sh
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/nvcomp/nemo
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT gui
    OPTIONAL
)

# Install file manager integration installer script
install(FILES
    install_file_manager_integration.sh
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/nvcomp
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT gui
    OPTIONAL
)

# Install documentation
install(FILES
    README.md
    FILE_MANAGER_TESTING.md
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc/nvcomp/linux
    COMPONENT gui
    OPTIONAL
)

# Optional: Install Python unit tests (for development/testing)
if(BUILD_TESTING)
    install(FILES
        test_nautilus_extension.py
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/nvcomp/tests
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT gui
        OPTIONAL
    )
endif()


