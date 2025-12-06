cmake_minimum_required(VERSION 3.18)
project(NvCompCLI LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

#==== START get nvCOMP SDK
include(FetchContent)

FetchContent_Declare(
  nvcomp_archive
  URL https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/windows-x86_64/nvcomp-windows-x86_64-5.1.0.21_cuda13-archive.zip
)

FetchContent_GetProperties(nvcomp_archive)
if(NOT nvcomp_archive_POPULATED)
  message(STATUS "Downloading nvCOMP archive...")
  FetchContent_Populate(nvcomp_archive)
endif()

# Check if the archive extracted to a subdirectory or if the files are in the root
if(EXISTS "${nvcomp_archive_SOURCE_DIR}/include/nvcomp.hpp")
    set(NVCOMP_PATH "${nvcomp_archive_SOURCE_DIR}")
else()
    # If not in root, look for the expected subdirectory structure
    file(GLOB CHILDREN "${nvcomp_archive_SOURCE_DIR}/*")
    foreach(CHILD ${CHILDREN})
        if(IS_DIRECTORY "${CHILD}" AND EXISTS "${CHILD}/include/nvcomp.hpp")
            set(NVCOMP_PATH "${CHILD}")
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED NVCOMP_PATH)
    message(FATAL_ERROR "Could not determine NVCOMP_PATH from downloaded archive in ${nvcomp_archive_SOURCE_DIR}")
endif()

message(STATUS "Using nvCOMP at: ${NVCOMP_PATH}")
#==== DONE get nvCOMP SDK

# Add the cmake directory to the prefix path so find_package can work
list(APPEND CMAKE_PREFIX_PATH "${NVCOMP_PATH}/lib/cmake/nvcomp")

# Find nvCOMP package
find_package(nvcomp REQUIRED)

# Create the executable
add_executable(nvcomp_cli main.cu)

# Link against nvCOMP
# The imported target is typically nvcomp::nvcomp for the main library
target_link_libraries(nvcomp_cli PRIVATE nvcomp::nvcomp)

# Copy DLLs to the output directory for convenience on Windows
if(WIN32)
    add_custom_command(TARGET nvcomp_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_PATH}/bin/nvcomp64_5.dll"
        $<TARGET_FILE_DIR:nvcomp_cli>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NVCOMP_PATH}/bin/nvcomp_cpu64_5.dll"
        $<TARGET_FILE_DIR:nvcomp_cli>
    )
endif()

