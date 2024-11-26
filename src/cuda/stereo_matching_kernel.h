// include/stereo_matching_kernel.h

#ifndef STEREO_MATCHING_KERNEL_H
#define STEREO_MATCHING_KERNEL_H

// CUDA Kernel Declaration for Wavefront Parallelization for a Single DP Matrix
__global__ void fillMatrixWavefrontKernel(int* d_matrix, int rows, int cols, 
                                         const int* d_left, const int* d_right, 
                                         int matchScore, int mismatchPenalty, int gapPenalty, 
                                         int diagonal);

#endif // STEREO_MATCHING_KERNEL_H
