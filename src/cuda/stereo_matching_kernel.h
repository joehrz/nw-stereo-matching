// src/cuda/stereo_matching_kernel.h

#ifndef STEREO_MATCHING_KERNEL_H
#define STEREO_MATCHING_KERNEL_H

#include <cuda_runtime.h>

// Define window size for SAD
#define WINDOW_SIZE 5
#define HALF_WINDOW (WINDOW_SIZE / 2)

// CUDA kernel declaration
__global__ void disparity_kernel(const int* left_image, const int* right_image, int* disparity_map, int width, int height, int max_disparity);

#endif // STEREO_MATCHING_KERNEL_H
