// src/main.cpp

#include "stereo_matching.h"
#include <iostream>
#include <memory>   // For smart pointers (optional)
#include <cstring>  // For std::memset

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: stereo_matching <left_image> <right_image> <output_disparity>\n";
        return EXIT_FAILURE;
    }

    std::string left_image_path = argv[1];
    std::string right_image_path = argv[2];
    std::string output_disparity_path = argv[3];

    // Load images
    Image left_image = load_image(left_image_path);
    Image right_image = load_image(right_image_path);

    // Convert images to int arrays
    int* h_left = convert_to_int(left_image.data, left_image.width * left_image.height);
    int* h_right = convert_to_int(right_image.data, right_image.width * right_image.height);

    // Allocate score matrix
    int* h_score_matrix = new int[left_image.width * left_image.height];
    std::memset(h_score_matrix, 0, left_image.width * left_image.height * sizeof(int));

    // Compute disparity using CUDA
    compute_disparity_cuda(h_left, h_right, h_score_matrix, left_image.width, left_image.height, 16);

    // Extract disparity map
    int* disparity_map = extract_disparity(h_score_matrix, left_image.width, left_image.height, 16);

    // Save disparity map
    save_disparity_map(output_disparity_path, disparity_map, left_image.width, left_image.height);

    // Clean up
    delete[] h_left;
    delete[] h_right;
    delete[] h_score_matrix;
    delete[] disparity_map;
    delete[] left_image.data;
    delete[] right_image.data;

    std::cout << "Disparity map saved to " << output_disparity_path << std::endl;

    return EXIT_SUCCESS;
}

