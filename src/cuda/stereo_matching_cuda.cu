// src/cuda/stereo_matching_cuda.cu

#include "stereo_matching_kernel.h"
#include "stereo_matching.h" // Include the main header for Image struct and other functions
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Implement compute_disparity_cuda function
void compute_disparity_cuda(const int* left_image, const int* right_image, int* score_matrix, int width, int height, int max_disparity) {
    // Calculate the size of the images in bytes
    size_t image_size = width * height * sizeof(int);

    // Device pointers
    int* d_left_image = nullptr;
    int* d_right_image = nullptr;
    int* d_score_matrix = nullptr;

    // Allocate device memory
    cudaCheckError( cudaMalloc((void**)&d_left_image, image_size) );
    cudaCheckError( cudaMalloc((void**)&d_right_image, image_size) );
    cudaCheckError( cudaMalloc((void**)&d_score_matrix, image_size) );

    // Copy input images from host to device
    cudaCheckError( cudaMemcpy(d_left_image, left_image, image_size, cudaMemcpyHostToDevice) );
    cudaCheckError( cudaMemcpy(d_right_image, right_image, image_size, cudaMemcpyHostToDevice) );

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize( (width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y );

    // Launch the CUDA kernel
    disparity_kernel<<<gridSize, blockSize>>>(d_left_image, d_right_image, d_score_matrix, width, height, max_disparity);

    // Check for any errors launching the kernel
    cudaCheckError( cudaGetLastError() );

    // Wait for the GPU to finish
    cudaCheckError( cudaDeviceSynchronize() );

    // Copy the disparity map back to host
    cudaCheckError( cudaMemcpy(score_matrix, d_score_matrix, image_size, cudaMemcpyDeviceToHost) );

    // Free device memory
    cudaFree(d_left_image);
    cudaFree(d_right_image);
    cudaFree(d_score_matrix);
}


