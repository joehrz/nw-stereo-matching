// include/stereo_matching.h

#ifndef STEREO_MATCHING_H
#define STEREO_MATCHING_H

#include <vector>
#include <string>

// Structure to hold alignment results
struct AlignmentResult {
    int disparity;
    std::vector<int> leftAligned;   // Aligned left sequence
    std::vector<int> rightAligned;  // Aligned right sequence
};

// StereoMatcher class encapsulates the NW algorithm for stereo matching
class StereoMatcher {
public:
    // Constructor to initialize scoring parameters
    StereoMatcher(int matchScore, int mismatchPenalty, int gapPenalty);

    // CPU version
    AlignmentResult computeAlignment(const std::vector<int>& leftLine, const std::vector<int>& rightLine);

    // CUDA Wavefront Parallelization version
    AlignmentResult computeAlignmentCUDA(const std::vector<int>& leftLine, const std::vector<int>& rightLine);


private:
    int matchScore_;
    int mismatchPenalty_;
    int gapPenalty_;

    // Helper functions for CPU implementation
    void initializeMatrix(std::vector<int>& matrix, int rows, int cols);
    void fillMatrix(const std::vector<int>& left, const std::vector<int>& right, std::vector<int>& matrix, int rows, int cols);
    AlignmentResult backtrack(const std::vector<int>& left, const std::vector<int>& right, const std::vector<int>& matrix, int rows, int cols);

    // CUDA helper function
    void fillMatrixWavefrontCUDA(int* matrix, int rows, int cols, 
                                 const std::vector<int>& left, const std::vector<int>& right);
};

#endif // STEREO_MATCHING_H
