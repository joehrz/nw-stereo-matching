// src/cpu/stereo_matching.cpp

#include "stereo_matching.h"
#include <algorithm>

// Example: CPU-based disparity computation using a simple block matching approach
void compute_disparity_cpu(const unsigned char* left_image, const unsigned char* right_image, int* disparity_map, int width, int height, int max_disparity) {
    for(int y = 0; y < height; ++y){
        for(int x = 0; x < width; ++x){
            int best_disparity = 0;
            int min_ssd = INT32_MAX;
            for(int d = 0; d < max_disparity; ++d){
                if(x - d < 0) continue;
                int ssd = 0;
                // Simple sum of squared differences (SSD) over a small window
                int window_size = 5;
                int half_window = window_size / 2;
                for(int wy = -half_window; wy <= half_window; ++wy){
                    for(int wx = -half_window; wx <= half_window; ++wx){
                        int lx = x + wx;
                        int ly = y + wy;
                        int rx = x - d + wx;
                        int ry = y + wy;
                        if(lx < 0 || lx >= width || ly < 0 || ly >= height ||
                           rx < 0 || rx >= width || ry < 0 || ry >= height){
                            ssd += 255; // Penalize out-of-bounds
                            continue;
                        }
                        int diff = static_cast<int>(left_image[ly * width + lx]) - static_cast<int>(right_image[ry * width + rx]);
                        ssd += diff * diff;
                    }
                }
                if(ssd < min_ssd){
                    min_ssd = ssd;
                    best_disparity = d;
                }
            }
            disparity_map[y * width + x] = best_disparity;
        }
    }
}
