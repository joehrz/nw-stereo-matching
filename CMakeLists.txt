cmake_minimum_required(VERSION 3.20)

# CMP0135 deals with CUDA sysroot usage (CMake >= 3.22).
# By setting it to NEW, you enable the latest behavior if available.
cmake_policy(SET CMP0135 NEW)

project(NW_Stereo_Matching VERSION 1.0 LANGUAGES CXX CUDA)

# ------------------------------------------------------------------------------
# Global Settings
# ------------------------------------------------------------------------------

# Output binaries in <build>/bin/
set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# Set a unified MSVC runtime library (Dynamic /MD or /MDd).
# This ensures all targets use the same runtime linkage.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# ------------------------------------------------------------------------------
# CUDA Configuration
# ------------------------------------------------------------------------------

# Link the CUDA runtime statically (so you do not need CUDA DLLs at runtime).
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Target a specific GPU architecture. Adjust if your GPU differs from 8.6.
set(CMAKE_CUDA_ARCHITECTURES 86)

# ------------------------------------------------------------------------------
# C++ Standards
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# Find Dependencies
# ------------------------------------------------------------------------------

find_package(OpenCV REQUIRED)

# ------------------------------------------------------------------------------
# FetchContent for GoogleTest (optional, if you have tests)
# ------------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
# Force shared CRT for gtest on MSVC to match above setting
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------------------
# Source Files
# ------------------------------------------------------------------------------

set(CPU_SRC
    ${CMAKE_SOURCE_DIR}/src/cpu/stereo_matching.cpp
)

set(CUDA_SRC
    ${CMAKE_SOURCE_DIR}/src/cuda/stereo_matching_kernel.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/stereo_matching_cuda.cu
)

# ------------------------------------------------------------------------------
# CPU Library
# ------------------------------------------------------------------------------

add_library(cpu_lib STATIC ${CPU_SRC})
target_include_directories(cpu_lib
    PUBLIC
        # Use generator expressions for robust include paths
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ------------------------------------------------------------------------------
# CUDA Library
# ------------------------------------------------------------------------------

add_library(cuda_lib STATIC ${CUDA_SRC})
set_target_properties(cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(cuda_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ------------------------------------------------------------------------------
# Main Executable
# ------------------------------------------------------------------------------

add_executable(stereo_matching ${CMAKE_SOURCE_DIR}/src/main.cpp)

target_include_directories(stereo_matching PRIVATE
    # For main.cpp to include project headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    # Include OpenCV headers
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(stereo_matching
    PRIVATE
        cpu_lib
        cuda_lib
        ${OpenCV_LIBS}
)

set_target_properties(stereo_matching PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

