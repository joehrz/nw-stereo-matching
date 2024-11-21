// src/main.cpp

#include <iostream>
#include <numeric> // For std::accumulate
#include <opencv2/opencv.hpp>
#include "stereo_matching.h"

int main(int argc, char** argv) {
    if(argc != 4) {
        std::cerr << "Usage: ./stereo_matching [strategy] left_image.png right_image.png" << std::endl;
        std::cerr << "Strategies: cpu, cuda_wavefront" << std::endl;
        return -1;
    }

    std::string strategy = argv[1];
    std::string leftPath = argv[2];
    std::string rightPath = argv[3];

    // Load images in grayscale
    cv::Mat leftImg = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
    cv::Mat rightImg = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);

    if(leftImg.empty() || rightImg.empty()) {
        std::cerr << "Error loading images." << std::endl;
        return -1;
    }

    // Ensure images have the same number of rows (epipolar lines)
    if(leftImg.rows != rightImg.rows) {
        std::cerr << "Left and right images must have the same number of rows (epipolar lines)." << std::endl;
        return -1;
    }

    // Load ground truth disparity map
    cv::Mat gtDisparity = cv::imread("C:\\Users\\joe_h\\Desktop\\nw-stereo-matching\\data\\Tsukuba\\tsukuba_truth.jpg", cv::IMREAD_GRAYSCALE);
    if(gtDisparity.empty()) {
        std::cerr << "Error loading ground truth disparity map." << std::endl;
        return -1;
    }

    // Adjust for scaling factor
    double scalingFactor = 16.0; // Adjust if necessary
    cv::Mat gtDisparityFloat;
    gtDisparity.convertTo(gtDisparityFloat, CV_32F, 1.0 / scalingFactor);


    // Select an epipolar line (row). For demonstration, choose the middle row
    int row = leftImg.rows / 2;

    // Extract scanlines from images
    std::vector<int> leftLine(leftImg.cols);
    std::vector<int> rightLine(rightImg.cols);
    for(int col = 0; col < leftImg.cols; ++col) {
        leftLine[col] = static_cast<int>(leftImg.at<uchar>(row, col));
    }
    for(int col = 0; col < rightImg.cols; ++col) {
        rightLine[col] = static_cast<int>(rightImg.at<uchar>(row, col));
    }

    // Initialize StereoMatcher with scoring parameters
    int matchScore = 5;
    int mismatchPenalty = -3;
    int gapPenalty = -4;
    StereoMatcher matcher(matchScore, mismatchPenalty, gapPenalty);

    // Compute alignment based on selected strategy
    AlignmentResult result;
    if(strategy == "cpu") {
        result = matcher.computeAlignment(leftLine, rightLine);
    }
    else if(strategy == "cuda_wavefront") {
        result = matcher.computeAlignmentCUDA(leftLine, rightLine);
    }
    else {
        std::cerr << "Unknown strategy: " << strategy << std::endl;
        std::cerr << "Available strategies: cpu, cuda_wavefront" << std::endl;
        return -1;
    }

    // Extract per-pixel disparities from the alignment result
    std::vector<float> estimatedDisparities(leftImg.cols, -1.0f);

    size_t idxLeftPos = 0;   // Position in the original left sequence
    size_t idxRightPos = 0;  // Position in the original right sequence

    for(size_t idx = 0; idx < result.leftAligned.size(); ++idx) {
        if(result.leftAligned[idx] != -1 && result.rightAligned[idx] != -1) {
            // Match or mismatch
            int disparity = static_cast<int>(idxRightPos) - static_cast<int>(idxLeftPos);
            if(idxLeftPos < estimatedDisparities.size()) {
                estimatedDisparities[idxLeftPos] = static_cast<float>(disparity);
            }
            idxLeftPos++;
            idxRightPos++;
        }
        else {
            // Handle gaps
            if(result.leftAligned[idx] != -1) {
                idxLeftPos++;
            }
            if(result.rightAligned[idx] != -1) {
                idxRightPos++;
            }
        }
    }

    // Extract ground truth disparities for the selected scanline
    std::vector<float> gtDisparities(leftImg.cols, -1.0f);
    for(int col = 0; col < leftImg.cols; ++col) {
        gtDisparities[col] = gtDisparityFloat.at<float>(row, col);
    }

    // Compare estimated disparities with ground truth
    float totalError = 0.0f;
    int validPixelCount = 0;

    for(size_t col = 0; col < estimatedDisparities.size(); ++col) {
        float estDisp = estimatedDisparities[col];
        float gtDisp = gtDisparities[col];

        if(estDisp >= 0 && gtDisp > 0) { // Exclude invalid disparities
            float error = std::abs(estDisp - gtDisp);
            totalError += error;
            validPixelCount++;
        }
    }



    // Extract ground truth disparities at the row
    cv::Mat gtDisparityRow = gtDisparityFloat.row(row);
    std::vector<float> gtDisparityValues;
    for (int col = 0; col < gtDisparityRow.cols; ++col) {
        float disp = gtDisparityRow.at<float>(0, col);
        if (disp > 0) { // Exclude invalid disparities
            gtDisparityValues.push_back(disp);
        }
    }

    // Compute mean ground truth disparity
    float meanGtDisparity = 0.0f;
    if (!gtDisparityValues.empty()) {
        meanGtDisparity = std::accumulate(gtDisparityValues.begin(), gtDisparityValues.end(), 0.0f) / gtDisparityValues.size();
    }


    if(validPixelCount > 0) {
        float meanError = totalError / validPixelCount;
        std::cout << "Estimated Disparity at row " << row << ": " << result.disparity << std::endl;
        std::cout << "Ground Truth Mean Disparity at row " << row << ": " << meanGtDisparity << std::endl;
        std::cout << "Mean Absolute Error at row " << row << ": " << meanError << std::endl;
    } else {
        std::cout << "No valid pixels to compare at row " << row << "." << std::endl;
    }

    return 0;
}

