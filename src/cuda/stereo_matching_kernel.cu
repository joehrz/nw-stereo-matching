// src/cuda/stereo_matching_kernel.cu

#include "stereo_matching_kernel.h"
#include <algorithm>

// CUDA Kernel for Wavefront Parallelization
__global__ void fillMatrixWavefrontKernel(int* d_matrix, int rows, int cols, 
                                         const int* d_left, const int* d_right, 
                                         int matchScore, int mismatchPenalty, int gapPenalty, 
                                         int diagonal) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= diagonal)
        return;

    // Calculate i and j based on the diagonal
    int i = min(diagonal, rows - 1) - idx;
    int j = diagonal - i;

    if(i <= 0 || j <= 0 || i >= rows || j >= cols)
        return;
    


    // Compute the index in the matrix
    int matrix_index = i * cols + j;
    int diag_index = (i - 1) * cols + (j - 1);
    int up_index = (i - 1) * cols + j;
    int left_index = i * cols + (j - 1);

     // Compute match/mismatch
    int match = (d_left[i - 1] == d_right[j - 1]) ? matchScore : mismatchPenalty;

    // Calculate scores
    int scoreDiag = d_matrix[diag_index] + match;
    int scoreUp = d_matrix[up_index] + gapPenalty;
    int scoreLeft = d_matrix[left_index] + gapPenalty;





    // Update the matrix with the maximum score
    d_matrix[matrix_index] = max(max(scoreDiag, scoreUp), scoreLeft);
}