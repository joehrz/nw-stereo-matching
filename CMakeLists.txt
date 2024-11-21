cmake_minimum_required(VERSION 3.20)

# ========================================================================
# CMake Policies
# ========================================================================

cmake_policy(SET CMP0135 NEW)

# ========================================================================
# Project Declaration
# ========================================================================

project(NW_Stereo_Matching VERSION 1.0 LANGUAGES CXX CUDA)

# ========================================================================
# Output Directories
# ========================================================================

set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# ========================================================================
# MSVC Runtime Library Settings (Windows Specific)
# ========================================================================
# MSVC Runtime Library Settings
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ========================================================================
# CUDA Settings
# ========================================================================

# Use the static CUDA runtime library
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Specify CUDA architectures based on your GPU's compute capability
set(CMAKE_CUDA_ARCHITECTURES 86)

# ========================================================================
# C++ Standards
# ========================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ========================================================================
# Find Packages
# ========================================================================

# Find Packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ========================================================================
# Fetch Packages
# ========================================================================

# Fetch Packages
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ========================================================================
# Source Files
# ========================================================================

# Define CPU source files
set(CPU_SRC
    ${CMAKE_SOURCE_DIR}/src/cpu/stereo_matching.cpp
)

# Define CUDA source files
set(CUDA_SRC
    ${CMAKE_SOURCE_DIR}/src/cuda/stereo_matching_kernel.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/stereo_matching_cuda.cu
)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# ========================================================================
# Create CPU Static Library
# ========================================================================

add_library(cpu_lib STATIC ${CPU_SRC})

# Include directories for the CPU library
target_include_directories(cpu_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)


# ========================================================================
# Create CUDA Static Library
# ========================================================================

# Create CUDA Library
add_library(cuda_lib STATIC ${CUDA_SRC})
set_target_properties(cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(cuda_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Set MSVC Runtime Library for cuda_lib
if(MSVC)
    set_target_properties(cuda_lib PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    )
endif()

# ========================================================================
# Define Executable
# ========================================================================

# Define Executable
add_executable(stereo_matching ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_include_directories(stereo_matching PRIVATE
    ${CMAKE_SOURCE_DIR}/include
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

# Set MSVC Runtime Library for stereo_matching
if(MSVC)
    set_target_properties(stereo_matching PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    )
endif()

# ========================================================================
# Enable Testing
# ========================================================================

enable_testing()
add_subdirectory(tests)
