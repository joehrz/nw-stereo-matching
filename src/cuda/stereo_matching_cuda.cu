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
    size_t matrixSize = rows * cols * sizeof(int);
    size_t leftSize = (rows - 1) * sizeof(int);  // Each row has (rows -1) pixels
    size_t rightSize = (cols - 1) * sizeof(int); // Each row has (cols -1) pixels

    int* d_matrix;
    int* d_left;
    int* d_right;

    // Allocate device memory
    cudaCheckError(cudaMalloc(&d_matrix, matrixSize));
    cudaCheckError(cudaMalloc(&d_left, leftSize));
    cudaCheckError(cudaMalloc(&d_right, rightSize));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_left, left.data(), leftSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_right, right.data(), rightSize, cudaMemcpyHostToDevice));

    // Define CUDA kernel parameters
    int threadsPerBlock = 256;

    // Iterate over each diagonal
    for(int diagonal = 1; diagonal < (rows + cols - 1); ++diagonal) {
        // Calculate number of elements in this diagonal
        int numElements = 0;
        if(diagonal < rows)
            numElements = diagonal;
        else
            numElements = rows + cols - 1 - diagonal;

        // Calculate grid size based on number of elements in the diagonal
        int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        if(blocks == 0) blocks = 1;

        // Launch kernel for this diagonal
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
    cudaCheckError(cudaMemcpy(matrix, d_matrix, matrixSize, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaCheckError(cudaFree(d_matrix));
    cudaCheckError(cudaFree(d_left));
    cudaCheckError(cudaFree(d_right));
}

void StereoMatcher::fillMatrixWavefrontCUDA_Fused(int* matrix, int rows, int cols, 
                                                  const std::vector<int>& left, const std::vector<int>& right) {
    size_t matrixSize = rows * cols * sizeof(int);
    size_t leftSize = (rows - 1) * sizeof(int);
    size_t rightSize = (cols - 1) * sizeof(int);

    int* d_matrix;
    int* d_left;
    int* d_right;

    cudaCheckError(cudaMalloc(&d_matrix, matrixSize));
    cudaCheckError(cudaMalloc(&d_left, leftSize));
    cudaCheckError(cudaMalloc(&d_right, rightSize));

    cudaCheckError(cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_left, left.data(), leftSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_right, right.data(), rightSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int totalDiagonals = rows + cols - 1;
    int groupSize = 10;  // Fuse 10 diagonals per kernel launch.
    for (int startDiag = 1; startDiag < totalDiagonals; startDiag += groupSize) {
        int endDiag = std::min(startDiag + groupSize - 1, totalDiagonals - 1);
        // Launch fused kernel for a group of diagonals.
        fusedFillMatrixKernel<<<1, threadsPerBlock>>>(d_matrix, rows, cols,
                                                       d_left, d_right,
                                                       gapPenalty_, startDiag, endDiag);
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());
    }

    cudaCheckError(cudaMemcpy(matrix, d_matrix, matrixSize, cudaMemcpyDeviceToHost));

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

// Fused CUDA Wavefront Parallelization Alignment
AlignmentResult StereoMatcher::computeAlignmentCUDA_Fused(const std::vector<int>& leftLine, const std::vector<int>& rightLine) {
    int rows = leftLine.size() + 1;
    int cols = rightLine.size() + 1;
    std::vector<int> matrix(rows * cols, 0);

    // Initialize boundaries.
    for (int i = 0; i < rows; ++i) {
        matrix[i * cols] = i * gapPenalty_;
    }
    for (int j = 0; j < cols; ++j) {
        matrix[j] = j * gapPenalty_;
    }

    fillMatrixWavefrontCUDA_Fused(matrix.data(), rows, cols, leftLine, rightLine);
    return backtrack(leftLine, rightLine, matrix, rows, cols);
}