// src/cuda/stereo_matching_cuda.cu

#include "stereo_matching.h"          // Include the StereoMatcher class definition
#include "stereo_matching_kernel.h"   // Include the CUDA kernel declarations
#include <cuda_runtime.h>
#include <iostream>
#include <vector>      // Required for std::vector
#include <algorithm>   // Required for std::min

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Fill the DP matrix using Wavefront Parallelization
void StereoMatcher::fillMatrixWavefrontCUDA(int* matrix_ptr, int rows, int cols, 
                                           const std::vector<int>& left, const std::vector<int>& right) {
    // Note: 'matrix_ptr' is the host pointer to the matrix data.
    // 'rows' and 'cols' are dimensions of the DP matrix (left.size()+1, right.size()+1)
    
    size_t matrixSize = static_cast<size_t>(rows) * cols * sizeof(int);
    // Size of sequences (left.size() elements, right.size() elements)
    size_t leftSeqSize = left.size() * sizeof(int);   
    size_t rightSeqSize = right.size() * sizeof(int); 

    int* d_matrix;
    int* d_left;
    int* d_right;

    // Allocate device memory
    cudaCheckError(cudaMalloc(&d_matrix, matrixSize));
    cudaCheckError(cudaMalloc(&d_left, leftSeqSize));
    cudaCheckError(cudaMalloc(&d_right, rightSeqSize));

    // Copy data to device
    // The host 'matrix_ptr' already contains initialized boundaries.
    cudaCheckError(cudaMemcpy(d_matrix, matrix_ptr, matrixSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_left, left.data(), leftSeqSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_right, right.data(), rightSeqSize, cudaMemcpyHostToDevice));

    // Define CUDA kernel parameters
    int threadsPerBlock = 256; // Typical value, can be tuned

    // Iterate over each anti-diagonal for the inner part of the matrix
    // The 'diagonal' parameter for the kernel corresponds to k-1 where k = i_mat + j_mat (1-based matrix indices)
    // or sum of 0-based sequence indices.
    // The loop for 'diagonal' should go from 1 up to (rows - 1) + (cols - 1) - 1 = rows + cols - 3
    // if 'diagonal' in kernel is sum of 0-based sequence indices.
    // Original loop: for(int diagonal = 1; diagonal < (rows + cols - 1); ++diagonal)
    // This 'diagonal' seems to be the k = (i_mat-1) + (j_mat-1) + 1 where i_mat, j_mat are 1-based matrix indices.
    // Or more simply, it's the "wavefront number".
    // The kernel calculates for matrix cells M[i][j] where i,j >= 1.
    for(int k = 1; k < (rows -1 + cols -1) ; ++k) { // k is the sum of (0-based) sequence indices being compared
                                                 // i.e. for M[i_m][j_m], this k = (i_m-1) + (j_m-1)
                                                 // So k goes from 0 (for M[1][1]) to (rows-2)+(cols-2)
                                                 // The kernel's 'diagonal' parameter seems to be this 'k' +1
                                                 // Let's use the original loop structure for 'diagonal' for consistency
                                                 // with how the kernel might interpret it.
        int current_diagonal_param_for_kernel = k; // This is what the kernel expects as 'diagonal'

        // Calculate number of elements on this diagonal to set grid size
        // This logic needs to be robust for calculating num_elements for the kernel's 'diagonal' interpretation
        int numElements = 0;
        // This is a simplified way to get an upper bound for elements on an anti-diagonal
        // A more precise calculation based on kernel's 'i_start' and 'i_end' is better.
        if (current_diagonal_param_for_kernel < std::min(rows - 1, cols - 1)) {
            numElements = current_diagonal_param_for_kernel;
        } else if (current_diagonal_param_for_kernel < std::max(rows - 1, cols - 1)) {
            numElements = std::min(rows - 1, cols - 1);
        } else {
            numElements = (rows - 1) + (cols - 1) - current_diagonal_param_for_kernel;
        }
        if (numElements <=0) numElements = 1; // Ensure at least one block if logic is tricky

        int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        if(blocks == 0) blocks = 1; // Ensure at least one block

        fillMatrixWavefrontKernel<<<blocks, threadsPerBlock>>>(
            d_matrix, rows, cols, 
            d_left, d_right,
            matchScore_, mismatchPenalty_, gapPenalty_, // Pass new scores
            current_diagonal_param_for_kernel // This is the 'diagonal' parameter for the kernel
        );
        cudaCheckError(cudaGetLastError()); // Check for kernel launch errors
        cudaCheckError(cudaDeviceSynchronize()); // Synchronize after each diagonal
    }

    // Copy result back to host
    cudaCheckError(cudaMemcpy(matrix_ptr, d_matrix, matrixSize, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaCheckError(cudaFree(d_matrix));
    cudaCheckError(cudaFree(d_left));
    cudaCheckError(cudaFree(d_right));
}

void StereoMatcher::fillMatrixWavefrontCUDA_Fused(int* matrix_ptr, int rows, int cols, 
                                                  const std::vector<int>& left, const std::vector<int>& right) {
    size_t matrixSize = static_cast<size_t>(rows) * cols * sizeof(int);
    size_t leftSeqSize = left.size() * sizeof(int);
    size_t rightSeqSize = right.size() * sizeof(int);

    int* d_matrix;
    int* d_left;
    int* d_right;

    cudaCheckError(cudaMalloc(&d_matrix, matrixSize));
    cudaCheckError(cudaMalloc(&d_left, leftSeqSize));
    cudaCheckError(cudaMalloc(&d_right, rightSeqSize));

    cudaCheckError(cudaMemcpy(d_matrix, matrix_ptr, matrixSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_left, left.data(), leftSeqSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_right, right.data(), rightSeqSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256; // Can be tuned
    // Total number of anti-diagonals to compute for the inner matrix M[1..rows-1][1..cols-1]
    // The 'diagonal' parameter in the kernel refers to this.
    int totalKernelDiagonals = (rows - 1) + (cols - 1) - 1; // Max sum of 0-based seq indices + 1

    int groupSize = 10;  // Number of diagonals to fuse per kernel launch. Tune this.
                         // Ensure groupSize is reasonable compared to totalKernelDiagonals.
    
    // The 'startDiag' and 'endDiag' for fusedFillMatrixKernel should correspond to
    // the 'diagonal' parameter expected by the kernel logic.
    // If kernel's 'diagonal' is k (sum of 0-based sequence indices) + 1,
    // then it goes from 1 to (rows-2)+(cols-2)+1 = totalKernelDiagonals
    for (int diag_group_start = 1; diag_group_start <= totalKernelDiagonals; diag_group_start += groupSize) {
        int diag_group_end = std::min(diag_group_start + groupSize - 1, totalKernelDiagonals);
        
        if (diag_group_start > diag_group_end) continue; // Should not happen if loop condition is correct

        fusedFillMatrixKernel<<<1, threadsPerBlock>>>( // Launch 1 block, threads cooperate
            d_matrix, rows, cols,
            d_left, d_right,
            matchScore_, mismatchPenalty_, gapPenalty_, // Pass new scores
            diag_group_start, diag_group_end
        );
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize()); // Synchronize after each fused group
    }

    cudaCheckError(cudaMemcpy(matrix_ptr, d_matrix, matrixSize, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_matrix));
    cudaCheckError(cudaFree(d_left));
    cudaCheckError(cudaFree(d_right));
}

// CUDA Wavefront Parallelization Alignment
AlignmentResult StereoMatcher::computeAlignmentCUDA(const std::vector<int>& leftLine, const std::vector<int>& rightLine) {
    int rows = leftLine.size() + 1;
    int cols = rightLine.size() + 1;
    std::vector<int> matrix_data(static_cast<size_t>(rows) * cols, 0); // Use vector for host memory

    // Initialize matrix boundaries on host
    initializeMatrix(matrix_data, rows, cols); // Use the common initialization logic

    // Fill the matrix using CUDA Wavefront Parallelization
    fillMatrixWavefrontCUDA(matrix_data.data(), rows, cols, leftLine, rightLine);

    // Perform backtracking on the CPU (using the matrix_data filled by CUDA)
    return backtrack(leftLine, rightLine, matrix_data, rows, cols);
}

// Fused CUDA Wavefront Parallelization Alignment
AlignmentResult StereoMatcher::computeAlignmentCUDA_Fused(const std::vector<int>& leftLine, const std::vector<int>& rightLine) {
    int rows = leftLine.size() + 1;
    int cols = rightLine.size() + 1;
    std::vector<int> matrix_data(static_cast<size_t>(rows) * cols, 0);

    // Initialize boundaries.
    initializeMatrix(matrix_data, rows, cols);

    fillMatrixWavefrontCUDA_Fused(matrix_data.data(), rows, cols, leftLine, rightLine);
    
    return backtrack(leftLine, rightLine, matrix_data, rows, cols);
}