# tests/CMakeLists.txt

# Enable testing for this directory
enable_testing()

# Add the test executable
add_executable(test_stereo_matching unit/test_stereo_matching.cpp)

# Set output directory for the test executable
set_target_properties(test_stereo_matching PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
)

# Link the test executable against Google Test and the library
# Use correct target names: GTest::gtest and GTest::gtest_main
target_link_libraries(test_stereo_matching PRIVATE GTest::gtest GTest::gtest_main stereo_matching_lib)

# Enable C++17 for the test executable
target_compile_features(test_stereo_matching PRIVATE cxx_std_17)

# Add a test to CTest
add_test(NAME StereoMatchingUnitTests COMMAND test_stereo_matching)


