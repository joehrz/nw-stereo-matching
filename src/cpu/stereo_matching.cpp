// src/cpu/stereo_matching.cpp

#include "stereo_matching.h"
#include <algorithm>

// Constructor
StereoMatcher::StereoMatcher(int matchScore, int mismatchPenalty, int gapPenalty)
    : matchScore_(matchScore), mismatchPenalty_(mismatchPenalty), gapPenalty_(gapPenalty) {}

// CPU Alignment Computation
AlignmentResult StereoMatcher::computeAlignment(const std::vector<int>& leftLine, const std::vector<int>& rightLine) {
    int rows = leftLine.size() + 1;
    int cols = rightLine.size() + 1;
    std::vector<int> matrix(rows * cols, 0);

    // Initialize matrix boundaries
    initializeMatrix(matrix, rows, cols);

    // Fill the matrix
    fillMatrix(leftLine, rightLine, matrix, rows, cols);

    // Perform backtracking to get alignment
    return backtrack(leftLine, rightLine, matrix, rows, cols);
}

// Initialize DP matrix boundaries
void StereoMatcher::initializeMatrix(std::vector<int>& matrix, int rows, int cols) {
    for(int i = 0; i < rows; ++i) {
        matrix[i * cols] = i * gapPenalty_;
    }
    for(int j = 0; j < cols; ++j) {
        matrix[j] = j * gapPenalty_;
    }
}

// Fill DP matrix
void StereoMatcher::fillMatrix(const std::vector<int>& left, const std::vector<int>& right, std::vector<int>& matrix, int rows, int cols) {
    for(int i = 1; i < rows; ++i) {
        for(int j = 1; j < cols; ++j) {
            //int match = (left[i - 1] == right[j - 1]) ? matchScore_ : mismatchPenalty_;
            //////////////////////////////////////////////////////////////////////
            int intensityDiff = std::abs(left[i - 1] - right[j - 1]);
            int truncatedDiff = std::min(intensityDiff, 20);
            int match = - truncatedDiff;
            //////////////////////////////////////////////////////////////////////
            int scoreDiag = matrix[(i - 1) * cols + (j - 1)] + match;
            int scoreUp = matrix[(i - 1) * cols + j] + gapPenalty_;
            int scoreLeft = matrix[i * cols + (j - 1)] + gapPenalty_;
            matrix[i * cols + j] = std::max({scoreDiag, scoreUp, scoreLeft});
        }
    }
}

// Backtracking to determine disparity
AlignmentResult StereoMatcher::backtrack(const std::vector<int>& left, const std::vector<int>& right,
                                           const std::vector<int>& matrix, int rows, int cols) {
    int i = rows - 1;
    int j = cols - 1;

    std::vector<int> leftAligned;
    std::vector<int> rightAligned;

    while(i > 0 || j > 0) {
        int current = matrix[i * cols + j];
        bool moved = false;

        // Diagonal move: use intensity difference cost
        if(i > 0 && j > 0) {
            int intensityDiff = std::abs(left[i - 1] - right[j - 1]);
            int truncatedDiff = std::min(intensityDiff, 20);
            if(current == matrix[(i - 1) * cols + (j - 1)] - truncatedDiff) {
                leftAligned.push_back(left[i - 1]);
                rightAligned.push_back(right[j - 1]);
                i--;
                j--;
                moved = true;
            }
        }
        // Up move (gap in right)
        if(!moved && i > 0 && current == matrix[(i - 1) * cols + j] + gapPenalty_) {
            leftAligned.push_back(left[i - 1]);
            rightAligned.push_back(-1); // gap indicator
            i--;
            moved = true;
        }
        // Left move (gap in left)
        if(!moved && j > 0 && current == matrix[i * cols + (j - 1)] + gapPenalty_) {
            leftAligned.push_back(-1); // gap indicator
            rightAligned.push_back(right[j - 1]);
            j--;
            moved = true;
        }
        if(!moved) {
            // If none of the above conditions met (should rarely happen), break out.
            break;
        }
    }

    // Reverse sequences since we built them backwards
    std::reverse(leftAligned.begin(), leftAligned.end());
    std::reverse(rightAligned.begin(), rightAligned.end());

    // Optionally, compute a combined disparity (e.g., average of differences)
    // Here, we simply store the first encountered disparity.
    AlignmentResult result;
    result.leftAligned = leftAligned;
    result.rightAligned = rightAligned;
    // You might choose to compute result.disparity as the average shift:
    int sumDisp = 0, count = 0;
    size_t idxLeftPos = 0, idxRightPos = 0;
    for(size_t idx = 0; idx < leftAligned.size(); ++idx) {
        if(leftAligned[idx] != -1 && rightAligned[idx] != -1) {
            sumDisp +=  (static_cast<int>(idxLeftPos) - static_cast<int>(idxRightPos));
            //sumDisp += (static_cast<int>(idxRightPos) - static_cast<int>(idxLeftPos));
            count++;
            idxLeftPos++;
            idxRightPos++;
        } else {
            if(leftAligned[idx] != -1)
                idxLeftPos++;
            if(rightAligned[idx] != -1)
                idxRightPos++;
        }
    }
    result.disparity = (count > 0) ? sumDisp / count : 0;
    return result;
}
