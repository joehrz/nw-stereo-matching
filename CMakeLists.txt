# nw-stereo-matching/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# ========================================================================
# CMake Policies
# ========================================================================

# CMP0135 relates to the MSVC runtime library discovery
# Setting it to NEW ensures CMake uses the CMAKE_MSVC_RUNTIME_LIBRARY variable
cmake_policy(SET CMP0135 NEW)

# ========================================================================
# Project Declaration
# ========================================================================

# Define the project with CXX and CUDA languages
project(NW_Stereo_Matching VERSION 1.0 LANGUAGES CXX CUDA)

# ========================================================================
# Output Directories
# ========================================================================

# Set the output directory for executables and libraries
set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

# ========================================================================
# MSVC Runtime Library Settings (Windows Specific)
# ========================================================================

# Ensure consistent runtime libraries to avoid linker warnings (LNK4098)
if(MSVC)
    # Options:
    # MultiThreadedDLL corresponds to /MD and /MDd (Dynamic)
    # MultiThreaded corresponds to /MT and /MTd (Static)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ========================================================================
# CUDA Settings
# ========================================================================

# Set CUDA runtime library type
# Changed from Static to Shared to match the C++ runtime library and resolve LNK4098
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Specify CUDA architectures based on your GPU's compute capability
# 'native' allows CMake to automatically detect and use the architecture of the host system's GPU
set(CMAKE_CUDA_ARCHITECTURES native)

# ========================================================================
# C++ Standards
# ========================================================================

# Set C++ standard for C++ and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ========================================================================
# CUDA Separable Compilation
# ========================================================================

# Enable CUDA separable compilation to allow separate compilation of device code
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ========================================================================
# Find Packages
# ========================================================================

# Find OpenCV package (assumes OpenCV is installed on the system)
find_package(OpenCV REQUIRED)

# ========================================================================
# Fetch Packages
# ========================================================================

# Include FetchContent module to download external dependencies like Google Test
include(FetchContent)

# Declare and download Google Test
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Make Google Test available
FetchContent_MakeAvailable(googletest)

# Verify GTest targets are available
if(TARGET GTest::gtest)
    message(STATUS "GTest::gtest is available")
else()
    message(FATAL_ERROR "GTest::gtest is not available")
endif()

if(TARGET GTest::gtest_main)
    message(STATUS "GTest::gtest_main is available")
else()
    message(FATAL_ERROR "GTest::gtest_main is not available")
endif()

# ========================================================================
# Source Files
# ========================================================================

# List of CPU source files
set(CPU_SRC
    src/cpu/stereo_matching.cpp
    # Add other CPU source files here
)

# List of CUDA source files
set(CUDA_SRC
    src/cuda/stereo_matching_kernel.cu
    src/cuda/stereo_matching_cuda.cu  # Added the host function CUDA file
    # Add other CUDA source files here
)

# Library source files
set(LIB_SRC
    src/lib/stereo_matching_lib.cpp
)

# ========================================================================
# Define Library
# ========================================================================

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Define a SHARED library target including CPU and CUDA source files
add_library(stereo_matching_lib SHARED ${LIB_SRC} ${CPU_SRC} ${CUDA_SRC})

# Set output directory for the library
set_target_properties(stereo_matching_lib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
)

# ========================================================================
# Include Directories and Link Libraries for the Library
# ========================================================================

# Specify include directories for the library
target_include_directories(stereo_matching_lib PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(stereo_matching_lib PUBLIC
    ${OpenCV_LIBS}
)

# ========================================================================
# Set CUDA Properties for the Library
# ========================================================================

# Set CUDA-specific properties for the library
set_target_properties(stereo_matching_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# ========================================================================
# Define Executable
# ========================================================================

# Define the executable target
add_executable(stereo_matching src/main.cpp)

# Set output directory for the executable
set_target_properties(stereo_matching PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
)

# ========================================================================
# Include Directories and Link Libraries for the Executable
# ========================================================================

# Specify include directories for the executable
target_include_directories(stereo_matching PRIVATE 
    include
)

# Link the executable against the library and OpenCV
target_link_libraries(stereo_matching PRIVATE 
    stereo_matching_lib 
    ${OpenCV_LIBS}
)

# ========================================================================
# Enable Testing
# ========================================================================

# Enable testing and add the tests subdirectory
enable_testing()
add_subdirectory(tests)
