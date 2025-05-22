// src/cpu/stereo_matching.cpp

#include "stereo_matching.h"
#include <algorithm> // For std::max, std::min, std::abs
#include <vector>    // For std::vector
#include <iostream>  // For debugging (optional)

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
        matrix[i * cols] = i * gapPenalty_; // Cost of gaps in right sequence
    }
    for(int j = 0; j < cols; ++j) {
        matrix[j] = j * gapPenalty_; // Cost of gaps in left sequence
    }
    // matrix[0][0] is already 0 due to the loops, or could be explicitly set.
}

// Fill DP matrix
void StereoMatcher::fillMatrix(const std::vector<int>& left, const std::vector<int>& right, std::vector<int>& matrix, int rows, int cols) {
    for(int i = 1; i < rows; ++i) {
        for(int j = 1; j < cols; ++j) {
            int intensityDiff = std::abs(left[i - 1] - right[j - 1]);
            int current_match_score_component;

            if (intensityDiff == 0) {
                current_match_score_component = matchScore_; // Use matchScore_ for perfect matches
            } else {
                int truncatedDiff = std::min(intensityDiff, 20); // Cap the difference penalty
                current_match_score_component = mismatchPenalty_ - truncatedDiff; // Base mismatch penalty + intensity difference penalty
            }
            
            int scoreDiag = matrix[(i - 1) * cols + (j - 1)] + current_match_score_component;
            int scoreUp   = matrix[(i - 1) * cols + j] + gapPenalty_; // Gap in right sequence
            int scoreLeft = matrix[i * cols + (j - 1)] + gapPenalty_; // Gap in left sequence
            
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
        int currentScore = matrix[i * cols + j];
        bool moved = false;

        // Check diagonal move first (match/mismatch)
        if (i > 0 && j > 0) {
            int intensityDiff = std::abs(left[i - 1] - right[j - 1]);
            int expected_score_component;
            if (intensityDiff == 0) {
                expected_score_component = matchScore_;
            } else {
                int truncatedDiff = std::min(intensityDiff, 20);
                expected_score_component = mismatchPenalty_ - truncatedDiff;
            }

            if (currentScore == matrix[(i - 1) * cols + (j - 1)] + expected_score_component) {
                leftAligned.push_back(left[i - 1]);
                rightAligned.push_back(right[j - 1]);
                i--;
                j--;
                moved = true;
            }
        }

        // Check upward move (gap in right sequence)
        if (!moved && i > 0) {
            if (currentScore == matrix[(i - 1) * cols + j] + gapPenalty_) {
                leftAligned.push_back(left[i - 1]);
                rightAligned.push_back(-1); // -1 indicates a gap
                i--;
                moved = true;
            }
        }
        
        // Check leftward move (gap in left sequence)
        if (!moved && j > 0) {
            if (currentScore == matrix[i * cols + (j - 1)] + gapPenalty_) {
                leftAligned.push_back(-1); // -1 indicates a gap
                rightAligned.push_back(right[j - 1]);
                j--;
                moved = true;
            }
        }

        if (!moved) {
            // This case should ideally not be reached if the matrix is filled correctly
            // and represents a valid path. Could happen if scores are ambiguous
            // or if i or j is 0 and the only option left was not taken above.
            // Force a move if stuck at a boundary.
            if (i > 0) { // Must have come from up
                leftAligned.push_back(left[i - 1]);
                rightAligned.push_back(-1);
                i--;
            } else if (j > 0) { // Must have come from left
                leftAligned.push_back(-1);
                rightAligned.push_back(right[j - 1]);
                j--;
            } else { // Both i and j are 0, break
                break;
            }
        }
    }

    std::reverse(leftAligned.begin(), leftAligned.end());
    std::reverse(rightAligned.begin(), rightAligned.end());

    AlignmentResult result;
    result.leftAligned = leftAligned;
    result.rightAligned = rightAligned;
    
    // Disparity calculation (e.g. for a single representative disparity) is often done
    // outside, based on the aligned sequences, or by analyzing the path.
    // For per-pixel disparity map, this is handled in main.cpp
    // Here, we can set a placeholder or a simple average for result.disparity if needed by tests.
    int sumDisp = 0, count = 0;
    size_t idxLeftPos = 0, idxRightPos = 0;
    for(size_t idx = 0; idx < leftAligned.size(); ++idx) {
        if(leftAligned[idx] != -1 && rightAligned[idx] != -1) {
            sumDisp += (static_cast<int>(idxLeftPos) - static_cast<int>(idxRightPos));
            count++;
        }
        if(leftAligned[idx] != -1) idxLeftPos++;
        if(rightAligned[idx] != -1) idxRightPos++;
    }
    result.disparity = (count > 0) ? (sumDisp / count) : 0;

    return result;
}