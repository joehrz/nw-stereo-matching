// src/lib/stereo_matching_lib.cpp

#include "stereo_matching.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <algorithm>

// Implement load_image function
Image load_image(const std::string& path) {
    Image img;
    cv::Mat cv_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (cv_img.empty()) {
        std::cerr << "Error loading image: " << path << std::endl;
        exit(EXIT_FAILURE);
    }
    img.width = cv_img.cols;
    img.height = cv_img.rows;
    img.data = new unsigned char[img.width * img.height];
    std::memcpy(img.data, cv_img.data, img.width * img.height * sizeof(unsigned char));
    return img;
}

// Implement save_disparity_map function
void save_disparity_map(const std::string& path, const int* disparity_map, int width, int height) {
    // Convert disparity map to CV_32S for normalization
    cv::Mat disparity_img(height, width, CV_32S, const_cast<int*>(disparity_map));

    // Normalize to [0, 255] for visualization
    cv::Mat disparity_norm;
    cv::normalize(disparity_img, disparity_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Save the normalized disparity map
    if (!cv::imwrite(path, disparity_norm)) {
        std::cerr << "Error saving disparity map to: " << path << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Implement extract_disparity function
int* extract_disparity(const int* score_matrix, int width, int height, int max_disparity){
    // Allocate memory for the disparity map and initialize to zero
    int* disparity = new int[width * height];
    std::fill(disparity, disparity + (width * height), 0);

    // Start from the bottom-right corner of the DP matrix
    int i = height - 1;
    int j = width - 1;

    while (i > 0 && j > 0){
        int current = score_matrix[i * width + j];
        int diagonal = score_matrix[(i - 1) * width + (j - 1)];
        int up = score_matrix[(i - 1) * width + j];
        int left = score_matrix[i * width + (j - 1)];

        // Determine the direction of movement based on the scores
        if (current == diagonal + 1){ 
            // Assuming a matching score of +1
            // Move diagonally: this indicates a match
            disparity[i * width + j] = 1; // Example disparity increment
            i--;
            j--;

        }

        else if (current == up -2){
            // Gap penalty for deletion
            // Move up: indicates a gap in the right image (horizontal shift)
            disparity[i * width + j] = 0; // Assigning 0 to indicate no disparity
            i--;
        }
        else if (current == left - 2){
            // Gap penalty for insertion
            // Move left: indicates a gap in the left image (no horizontal shift)
            disparity[i * width + j] = 0; // Assigning 0 to indicate no disparity
            j--;
        }
        else {
            // If no valid move is found, terminate the backtracking
            break;
        }
    }

    while (j > 0){
        disparity[0 * width + j] = 0;
        j--;
    }
    while (i > 0){
        disparity[i * width + 0] = 0;
        i--;
    }

    return disparity;
}

// Implement convert_to_int function
int* convert_to_int(const unsigned char* data, int size) {
    int* int_data = new int[size];
    for(int i = 0; i < size; ++i){
        int_data[i] = static_cast<int>(data[i]);
    }
    return int_data;
}

// Implement compute_disparity_cpu function (if needed)
// void compute_disparity_cpu(...) {
//     // CPU-based disparity computation
// }

