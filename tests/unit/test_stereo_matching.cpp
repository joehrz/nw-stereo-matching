// tests/unit/test_stereo_matching.cpp

#include "stereo_matching.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <cstring>

// Helper function to create a synthetic image
Image create_synthetic_image(int width, int height, unsigned char value){
    Image img;
    img.width = width;
    img.height = height;
    img.data = new unsigned char[width * height];
    std::memset(img.data, value, width * height * sizeof(unsigned char));

    return img;
}


// Test image Loading with a valid path
TEST(StereoMatchingTest, LoadImageValidPath){
    // Create a temporary synthetic image and save it
    std::string temp_path = "temp_test_image.png"; // Added semicolon
    int width = 100;
    int height = 100;
    unsigned char value = 150; // Mid-gray
    Image synthetic_img = create_synthetic_image(width, height, value);

    // Convert to OpenCV Mat and save
    cv::Mat cv_img(height, width, CV_8UC1, synthetic_img.data);
    bool success = cv::imwrite(temp_path, cv_img);
    ASSERT_TRUE(success) << "Failed to write temporary test image.";

    // Load the image using the load_image function
    Image loaded_img = load_image(temp_path);
    
    // Verify dimensions
    EXPECT_EQ(loaded_img.width, width);
    EXPECT_EQ(loaded_img.height, height);
    
    // Verify pixel values
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(loaded_img.data[i], value);
    }
    
    // Clean up
    delete[] synthetic_img.data;
    delete[] loaded_img.data;
    // Remove the temporary image file
    std::remove(temp_path.c_str());
}

// Test image Loading with an invalid path
TEST(StereoMatchingTest, LoadImageInvalidPath) {
    // Expect the program to exit when loading a non-existent image
    std::string invalid_path = "non_existent_image.png";
    
    // To test exit behavior, use ASSERT_DEATH (requires the test to be run in an environment that supports it)
    ASSERT_DEATH(load_image(invalid_path), "Error loading image");
}

// Test score matrix initialization
TEST(StereoMatchingTest, InitializeScoreMatrix) {
    // Define image dimensions
    int width = 5;
    int height = 5;
    int max_disparity = 16; // Not used in initialization
    
    // Create synthetic left and right images (values are irrelevant for initialization)
    Image left = create_synthetic_image(width, height, 100);
    Image right = create_synthetic_image(width, height, 100);

    // Convert to int arrays
    int* h_left = convert_to_int(left.data, width * height);
    int* h_right = convert_to_int(right.data, width * height);

    // Allocate and initialize score matrix
    int* score_matrix = new int[width * height];
    std::memset(score_matrix, 0, width * height * sizeof(int));

    // Compute disparity using the NW-based algorithm
    compute_disparity_cuda(h_left, h_right, score_matrix, width, height, max_disparity);

    // Verify the initialization of the first row and first column
    for (int j = 0; j < width; ++j) {
        EXPECT_EQ(score_matrix[0 * width + j], j * (-2)) << "Score matrix initialization failed at (0, " << j << ")";
    }
    
    for (int i = 0; i < height; ++i) {
        EXPECT_EQ(score_matrix[i * width + 0], i * (-2)) << "Score matrix initialization failed at (" << i << ", 0)";
    }
    
    // Clean up
    delete[] left.data;
    delete[] right.data;
    delete[] score_matrix;
    delete[] h_left;
    delete[] h_right;
}

// Test disparity extraction with a known score matrix
TEST(StereoMatchingTest, ExtractDisparityKnownScoreMatrix) {
    // Define a simple 3x3 score matrix with known values
    // Example score matrix:
    //  0  -2  -4
    // -2   1  -1
    // -4  -1   2
    int width = 3;
    int height = 3;
    int max_disparity = 16; // Not used in this test

    int score_matrix[9] = {
        0,  -2, -4,
        -2, 1,  -1,
        -4, -1, 2
    };
    
    // Expected disparity map after backtracking:
    // Assuming that a higher score indicates a better match, the optimal path would be (2,2) -> (1,1) -> (0,0)
    // Therefore, disparities at (2,2) and (1,1) should be 1, others 0
    int expected_disparity[9] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    
    // Extract disparity
    int* disparity = extract_disparity(score_matrix, width, height, max_disparity);

    // Verify disparity map
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(disparity[i * width + j], expected_disparity[i * width + j])
                << "Disparity extraction failed at (" << i << ", " << j << ")";
        }
    }
    
    // Clean up
    delete[] disparity;
}

// Test CUDA kernel correctness with a simple input
TEST(StereoMatchingTest, CudaKernelCorrectness) {
    // Define image dimensions
    int width = 3;
    int height = 3;
    int max_disparity = 16;
    
    // Create synthetic left and right images with a simple pattern
    // Left image:
    // 100 100 100
    // 100 100 100
    // 100 100 100
    // Right image:
    // 100 100 100
    // 100 100 100
    // 100 100 100
    Image left = create_synthetic_image(width, height, 100);
    Image right = create_synthetic_image(width, height, 100);

    // Convert to int arrays
    int* h_left = convert_to_int(left.data, width * height);
    int* h_right = convert_to_int(right.data, width * height);

    // Allocate and initialize score matrix
    int* score_matrix = new int[width * height];
    std::memset(score_matrix, 0, width * height * sizeof(int));

    // Compute disparity using the NW-based algorithm
    compute_disparity_cuda(h_left, h_right, score_matrix, width, height, max_disparity);
    
    // Expected score matrix:
    // 0  -2  -4
    // -2 2   0
    // -4 0   3
    int expected_score_matrix[9] = {
        0,  -2, -4,
        -2, 2,  0,
        -4, 0,  3
    };


    // Verify score matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            EXPECT_EQ(score_matrix[i * width + j], expected_score_matrix[i * width + j])
                << "CUDA kernel computation failed at (" << i << ", " << j << ")";
        }
    }
    
    // Clean up
    delete[] left.data;
    delete[] right.data;
    delete[] score_matrix;
    delete[] h_left;
    delete[] h_right;
}

// Test disparity map generation with synthetic data
TEST(StereoMatchingTest, GenerateDisparityMap) {
    // Define image dimensions
    int width = 4;
    int height = 4;
    int max_disparity = 16;
    
    // Create synthetic left and right images with a known shift
    // Left image:
    // 100 100 100 100
    // 100 100 100 100
    // 100 100 100 100
    // 100 100 100 100
    // Right image shifted right by 1 pixel:
    // 0   100 100 100
    // 0   100 100 100
    // 0   100 100 100
    // 0   100 100 100
    Image left = create_synthetic_image(width, height, 100);
    Image right = create_synthetic_image(width, height, 0); // Initialize to 0

    // Manually shift the left image to the right by 1 pixel to create the right image
    for (int i = 0; i < height; ++i){
        for (int j = 1; j < width; ++j){ // Changed 'i' to 'j'
            right.data[i * width + j] = left.data[i * width + (j - 1)];
        }
    }

    // Convert to int arrays
    int* h_left = convert_to_int(left.data, width * height);
    int* h_right = convert_to_int(right.data, width * height);

    // Allocate and initialize score matrix
    int* score_matrix = new int[width * height];
    std::memset(score_matrix, 0, width * height * sizeof(int));

    // Compute disparity using the NW-based algorithm
    compute_disparity_cuda(h_left, h_right, score_matrix, width, height, max_disparity);
    
    // Extract disparity
    int* disparity = extract_disparity(score_matrix, width, height, max_disparity);

    // Expected disparity map:
    // Since the right image is shifted right by 1, disparities should be 1 where the shift exists
    int expected_disparity[16] = {
        0, 0, 0, 0,
        0, 1, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 1
    };


    // Verify the disparity map
    for (int i = 0; i < height; ++i){
        for (int j = 0; j < width; ++j){
            EXPECT_EQ(disparity[i * width + j], expected_disparity[i * width + j])
                << "Disparity map generation failed at (" << i << ", " << j << ")";            
        }
    }

    // Clean up
    delete[] left.data;
    delete[] right.data;
    delete[] score_matrix;
    delete[] disparity;
    delete[] h_left;
    delete[] h_right;
}
