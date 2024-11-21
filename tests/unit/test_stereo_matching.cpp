// tests/unit/test_stereo_matching.cpp

#include <gtest/gtest.h>
#include "stereo_matching.h"

// Helper function to compare results across strategies
void compareStrategies(StereoMatcher& matcher, const std::vector<int>& leftLine, const std::vector<int>& rightLine) {
    AlignmentResult cpuResult = matcher.computeAlignment(leftLine, rightLine);
    AlignmentResult wavefrontResult = matcher.computeAlignmentCUDA(leftLine, rightLine);

    // All disparities should be equal
    EXPECT_EQ(cpuResult.disparity, wavefrontResult.disparity);
}

// Test case for identical lines
TEST(StereoMatcherTest, IdenticalLines) {
    std::vector<int> leftLine = {1, 2, 3, 4, 5};
    std::vector<int> rightLine = {1, 2, 3, 4, 5};

    StereoMatcher matcher(2, -1, -2);
    compareStrategies(matcher, leftLine, rightLine);
}

// Test case for shifted lines (disparity)
TEST(StereoMatcherTest, ShiftedLines) {
    std::vector<int> leftLine = {1, 2, 3, 4, 5};
    std::vector<int> rightLine = {0, 1, 2, 3, 4, 5};

    // Adjusted scoring parameters
    StereoMatcher matcher(2, -2, -1);

    AlignmentResult cpuResult = matcher.computeAlignment(leftLine, rightLine);
    AlignmentResult wavefrontResult = matcher.computeAlignmentCUDA(leftLine, rightLine);

    // Expected disparity is 1
    EXPECT_EQ(cpuResult.disparity, 1);
    EXPECT_EQ(wavefrontResult.disparity, 1);
}

// Test case for completely different lines
TEST(StereoMatcherTest, DifferentLines) {
    std::vector<int> leftLine = {1, 1, 1, 1, 1};
    std::vector<int> rightLine = {2, 2, 2, 2, 2};

    StereoMatcher matcher(2, -1, -2);
    compareStrategies(matcher, leftLine, rightLine);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
