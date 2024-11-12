// include/stereo_matching.h

#ifndef STEREO_MATCHING_H
#define STEREO_MATCHING_H

#include <string>

// Struct to hold image data
struct Image {
    int width;
    int height;
    unsigned char* data;
};

// Function declarations
Image load_image(const std::string& path);
void save_disparity_map(const std::string& path, const int* disparity_map, int width, int height);
int* extract_disparity(const int* score_matrix, int width, int height, int max_disparity);
void compute_disparity_cuda(const int* left_image, const int* right_image, int* score_matrix, int width, int height, int max_disparity);
int* convert_to_int(const unsigned char* data, int size);

#endif // STEREO_MATCHING_H
