// src/cuda/stereo_matching_kernel.cu

#include "stereo_matching_kernel.h"
#include <algorithm>

// CUDA Kernel for Wavefront Parallelization for a Single DP Matrix
__global__ void fillMatrixWavefrontKernel(int* d_matrix, int rows, int cols, 
                                            const int* d_left, const int* d_right, 
                                            int matchScore, int mismatchPenalty, int gapPenalty, 
                                            int diagonal) {
    // Determine valid row indices for this diagonal

    int d = diagonal;
    int i_start = max(1, d - (cols - 1));
    int i_end = min(d, rows - 1);
    int numElements = i_end - i_start + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements)
        return;

    // Map idx to (i, j) such that i + j = d + 1.
    int i = i_end - idx;        // i in descending order from i_end to i_start
    int j = d + 1 - i;          // ensures that i+j = d+1
    
    // Boundary check is now implicit (i in [1, rows-1], j in [1, cols-1])
    int matrix_index = i * cols + j;
    int diag_index = (i - 1) * cols + (j - 1);
    int up_index   = (i - 1) * cols + j;
    int left_index = i * cols + (j - 1);
    
    int intensityDiff = abs(d_left[i - 1] - d_right[j - 1]);
    int truncatedDiff = min(intensityDiff, 20);
    int scoreDiag = d_matrix[diag_index] - truncatedDiff;
    int scoreUp   = d_matrix[up_index] + gapPenalty;
    int scoreLeft = d_matrix[left_index] + gapPenalty;
    
    d_matrix[matrix_index] = max(max(scoreDiag, scoreUp), scoreLeft);
}
