// src/cuda/stereo_matching_kernel.cu

#include "stereo_matching_kernel.h"
#include <algorithm>

// CUDA Kernel for Wavefront Parallelization for a Single DP Matrix
__global__ void fillMatrixWavefrontKernel(int* d_matrix, int rows, int cols, 
                                         const int* d_left, const int* d_right, 
                                         int matchScore, int mismatchPenalty, int gapPenalty, 
                                         int diagonal) {
    // Each thread handles one cell in the current diagonal
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= diagonal)
        return;

    // Calculate i and j based on the diagonal and thread index
    int i = min(diagonal, rows - 1) - idx;
    int j = diagonal - i;

    // Boundary checks
    if(i <= 0 || j <= 0 || i >= rows || j >= cols)
        return;

    // Compute the index in the matrix for this cell
    int matrix_index = i * cols + j;
    int diag_index = (i - 1) * cols + (j - 1);
    int up_index = (i - 1) * cols + j;
    int left_index = i * cols + (j - 1);

    // Compute match/mismatch
    // Instead of (d_left[i - 1] == d_right[j - 1]) ? matchScore : mismatchPenalty
    // we define a difference-based cost:
    int intensityDiff = abs(d_left[i - 1] - d_right[j - 1]);
    int truncatedDiff = min(intensityDiff, 20);  // example threshold
    // Because the NW approach is "maximizing", we might interpret bigger difference as negative.
    int costFromDiag = d_matrix[diag_index] - truncatedDiff; 



    // Calculate scores
    int scoreDiag = costFromDiag;
    int scoreUp   = d_matrix[up_index]   + gapPenalty;
    int scoreLeft = d_matrix[left_index] + gapPenalty;

    d_matrix[matrix_index] = max(max(scoreDiag, scoreUp), scoreLeft);
}
