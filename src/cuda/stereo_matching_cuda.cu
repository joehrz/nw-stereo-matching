// src/cuda/stereo_matching_cuda.cu

#include "stereo_matching.h"          // Include the StereoMatcher class definition
#include "stereo_matching_kernel.h"   // Include the CUDA kernel declarations
#include <cuda_runtime.h>
#include <iostream>

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Fill the DP matrix using Wavefront Parallelization
void StereoMatcher::fillMatrixWavefrontCUDA(int* matrix, int rows, int cols, 
                                           const std::vector<int>& left, const std::vector<int>& right) {
    size_t size = rows * cols * sizeof(int);
    int* d_matrix;
    int* d_left;
    int* d_right;

    // Allocate device memory
    cudaCheckError(cudaMalloc(&d_matrix, size));
    cudaCheckError(cudaMalloc(&d_left, left.size() * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_right, right.size() * sizeof(int)));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_left, left.data(), left.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_right, right.data(), right.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Define CUDA kernel parameters
    int threadsPerBlock = 256;

    // Process diagonals starting from 1 to rows + cols - 2 (0-based indexing)
    for(int diagonal = 1; diagonal < (rows + cols - 1); ++diagonal) {
        // Calculate number of elements in this diagonal
        int numElements = 0;
        if(diagonal < rows)
            numElements = diagonal;
        else
            numElements = rows + cols - 1 - diagonal;

        // Calculate grid size
        int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        if(blocks == 0) blocks = 1;

        // Launch kernel
        fillMatrixWavefrontKernel<<<blocks, threadsPerBlock>>>(d_matrix, rows, cols, 
                                                               d_left, d_right,
                                                               matchScore_, mismatchPenalty_, gapPenalty_,
                                                               diagonal);
        // Check for kernel launch errors
        cudaCheckError(cudaGetLastError());

        // Synchronize to ensure completion before next diagonal
        cudaCheckError(cudaDeviceSynchronize());
    }

    // Copy result back to host
    cudaCheckError(cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaCheckError(cudaFree(d_matrix));
    cudaCheckError(cudaFree(d_left));
    cudaCheckError(cudaFree(d_right));
}


// CUDA Wavefront Parallelization Alignment
AlignmentResult StereoMatcher::computeAlignmentCUDA(const std::vector<int>& leftLine, const std::vector<int>& rightLine) {
    int rows = leftLine.size() + 1;
    int cols = rightLine.size() + 1;
    std::vector<int> matrix(rows * cols, 0);

    // Initialize matrix boundaries on host
    for(int i = 0; i < rows; ++i) {
        matrix[i * cols] = i * gapPenalty_;
    }
    for(int j = 0; j < cols; ++j) {
        matrix[j] = j * gapPenalty_;
    }

    // Fill the matrix using CUDA Wavefront Parallelization
    fillMatrixWavefrontCUDA(matrix.data(), rows, cols, leftLine, rightLine);

    // Perform backtracking on the CPU
    return backtrack(leftLine, rightLine, matrix, rows, cols);
}