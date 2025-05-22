// src/cuda/stereo_matching_kernel.cu

#include "stereo_matching_kernel.h"
#include <algorithm> // For __max, __min on device if not using CUDA's versions
#include <math.h>    // For abs

// CUDA Kernel for Wavefront Parallelization for a Single DP Matrix
__global__ void fillMatrixWavefrontKernel(int* d_matrix, int rows, int cols, 
                                            const int* d_left, const int* d_right, 
                                            int matchScore, int mismatchPenalty, int gapPenalty, 
                                            int diagonal) {
    // Determine valid row indices for this diagonal
    int d = diagonal; // Current anti-diagonal index (sum of i+j for matrix elements, adjusted)
                      // Or, if diagonal is k from 1 to rows+cols-2
                      // d is k in the loop from fillMatrixWavefrontCUDA

    // Calculate the range of 'i' indices for the current diagonal 'd'
    // This mapping ensures that threads only compute cells on the current wavefront.
    // The original mapping was a bit complex; simplifying for clarity:
    // For a diagonal 'diag_sum = i + j', where i and j are 1-based sequence indices.
    // The matrix indices are (i_mat = i, j_mat = j).
    // The kernel diagonal parameter 'diagonal' is likely the sum of matrix indices (0-based for matrix).
    // Let's assume 'diagonal' is the sum of 0-based matrix indices (i_m + j_m)
    // where i_m is from 1 to rows-1, and j_m is from 1 to cols-1.
    // So, 'diagonal' goes from 2 (for cell 1,1) to (rows-1) + (cols-1).
    // This kernel is launched for 'diagonal' from 1 to (rows + cols - 2) [as per original host code]
    // This means 'diagonal' parameter is likely k-1 from the definition of anti-diagonal k = i+j
    // where k goes from 1 to M+N-1.
    // The loop in fillMatrixWavefrontCUDA iterates 'diagonal' from 1 to (rows + cols - 1 -1)
    // Let's stick to the original kernel's indexing logic for 'diagonal' parameter.

    int i_start = max(1, diagonal - (cols - 1) + 1); // Adjusted for 1-based sequence access (i-1, j-1)
    int i_end = min(diagonal, rows - 1);             // Adjusted for 1-based sequence access

    int num_elements_on_diagonal = i_end - i_start + 1;
    if (num_elements_on_diagonal <= 0) return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements_on_diagonal) {
        return;
    }

    // Map thread index `idx` to matrix cell (i, j) on the current diagonal
    // i and j here are 1-based for matrix access (1 to rows-1, 1 to cols-1)
    int i = i_start + idx; 
    int j = diagonal + 1 - i; // Such that i+j = diagonal + 1 (for 1-based matrix indices)

    if (i <= 0 || i >= rows || j <= 0 || j >= cols) { // Boundary check for matrix indices
        return;
    }
    
    int matrix_index = i * cols + j; // Current cell M[i][j]
    
    // Scores from three possible previous cells
    // Diagonal: M[i-1][j-1]
    int intensityDiff = abs(d_left[i - 1] - d_right[j - 1]); // Access sequences with 0-based index
    int current_match_score_component;
    if (intensityDiff == 0) {
        current_match_score_component = matchScore;
    } else {
        int truncatedDiff = min(intensityDiff, 20); // Cap difference
        current_match_score_component = mismatchPenalty - truncatedDiff;
    }
    int scoreDiag = d_matrix[(i - 1) * cols + (j - 1)] + current_match_score_component;

    // Up: M[i-1][j]
    int scoreUp   = d_matrix[(i - 1) * cols + j] + gapPenalty;
    
    // Left: M[i][j-1]
    int scoreLeft = d_matrix[i * cols + (j - 1)] + gapPenalty;
    
    d_matrix[matrix_index] = max(max(scoreDiag, scoreUp), scoreLeft);
}


__global__ void fusedFillMatrixKernel(int* d_matrix, int rows, int cols,
                                        const int* d_left, const int* d_right,
                                        int matchScore, int mismatchPenalty, int gapPenalty, 
                                        int startDiag, int endDiag) {
    // This kernel processes a group of diagonals [startDiag, endDiag]
    // Each thread block (this kernel is launched with <<<1, threadsPerBlock>>>)
    // will iterate through its assigned cells within these diagonals.

    for (int d = startDiag; d <= endDiag; d++) { // Iterate over each diagonal in the fused group
        // Calculate the range of 'i' indices for the current diagonal 'd'
        // (similar logic to fillMatrixWavefrontKernel)
        int i_start = max(1, d - (cols - 1) + 1); 
        int i_end   = min(d, rows - 1);        
        int num_elements_on_diagonal = i_end - i_start + 1;

        if (num_elements_on_diagonal <= 0) {
             __syncthreads(); // Synchronize before next diagonal in the group
            continue;
        }
        
        // Parallelize computation of cells on the current diagonal 'd'
        for (int element_idx = threadIdx.x; element_idx < num_elements_on_diagonal; element_idx += blockDim.x) {
            int i = i_start + element_idx;
            int j = d + 1 - i;

            if (i <= 0 || i >= rows || j <= 0 || j >= cols) { // Should not happen if i_start/i_end are correct
                continue;
            }

            int matrix_index = i * cols + j;

            int intensityDiff = abs(d_left[i - 1] - d_right[j - 1]);
            int current_match_score_component;
            if (intensityDiff == 0) {
                current_match_score_component = matchScore;
            } else {
                int truncatedDiff = min(intensityDiff, 20);
                current_match_score_component = mismatchPenalty - truncatedDiff;
            }
            int scoreDiag = d_matrix[(i - 1) * cols + (j - 1)] + current_match_score_component;
            
            int scoreUp   = d_matrix[(i - 1) * cols + j] + gapPenalty;
            int scoreLeft = d_matrix[i * cols + (j - 1)] + gapPenalty;

            d_matrix[matrix_index] = max(max(scoreDiag, scoreUp), scoreLeft);
        }
        __syncthreads(); // IMPORTANT: Ensure all threads complete the current diagonal 'd'
                         // before any thread moves to the next diagonal 'd+1' in the fused group.
    }
}