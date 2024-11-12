// src/cuda/stereo_matching_kernel.cu

#include "stereo_matching_kernel.h"
#include <climits>
#include <cstdint>  // Added to define INT32_MAX
#include <cstdio>

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel for disparity computation using Sum of Absolute Differences (SAD)
__global__ void disparity_kernel(const int* left_image, const int* right_image, int* disparity_map, int width, int height, int max_disparity){
    // Calculate the x and y coordinates of the pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if(x >= width || y >= height){
        return;
    }

    int best_disparity = 0;
    int min_sad = INT32_MAX;

    // Iterate over all possible disparities
    for(int d = 0; d < max_disparity; ++d){
        // Ensure the shifted window stays within image boundaries
        if(x - d - HALF_WINDOW < 0){
            continue;
        }

        int sad = 0;

        // Compute SAD over the window
        for(int wy = -HALF_WINDOW; wy <= HALF_WINDOW; ++wy){
            for(int wx = -HALF_WINDOW; wx <= HALF_WINDOW; ++wx){
                int lx = x + wx;
                int ly = y + wy;
                int rx = lx - d;
                int ry = ly;

                // Boundary checks for window pixels
                if(lx < 0 || lx >= width || ly < 0 || ly >= height ||
                   rx < 0 || rx >= width || ry < 0 || ry >= height){
                    sad += 255; // Penalize out-of-bounds
                    continue;
                }

                int left_pixel = left_image[ly * width + lx];
                int right_pixel = right_image[ry * width + rx];
                sad += abs(left_pixel - right_pixel);
            }
        }

        // Update best disparity if current SAD is lower
        if(sad < min_sad){
            min_sad = sad;
            best_disparity = d;
        }
    }

    // Assign the best disparity to the disparity map
    disparity_map[y * width + x] = best_disparity;
}
