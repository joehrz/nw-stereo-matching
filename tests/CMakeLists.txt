# Define the test executable
add_executable(runUnitTests ${CMAKE_SOURCE_DIR}/tests/unit/test_stereo_matching.cpp)

# Include directories for the tests
target_include_directories(runUnitTests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link the test executable against libraries
target_link_libraries(runUnitTests
    PRIVATE
        cpu_lib
        cuda_lib
        GTest::gtest_main
        ${OpenCV_LIBS}
)

# Set properties for the test executable
set_target_properties(runUnitTests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set MSVC Runtime Library for runUnitTests
if(MSVC)
    set_target_properties(runUnitTests PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    )
endif()

# Register the test
add_test(NAME runUnitTests COMMAND runUnitTests)
