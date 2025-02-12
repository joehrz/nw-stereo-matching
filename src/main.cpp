#include <iostream>
#include <numeric> // For std::accumulate
#include <opencv2/opencv.hpp>
#include "stereo_matching.h"
#include <chrono> // For timing
#include <vector>
#include <omp.h> // For OpenMP parallelization

int main(int argc, char** argv) {
    if(argc != 5) {
        std::cerr << "Usage: ./stereo_matching [strategy] left_image.png right_image.png ground_truth.png" << std::endl;
        std::cerr << "Strategies: cpu, cuda_wavefront, cuda_wavefront_fused" << std::endl;
        return -1;
    }

    std::string strategy = argv[1];
    std::string leftPath = argv[2];
    std::string rightPath = argv[3];
    std::string gtPath = argv[4];

    // Load images in grayscale
    cv::Mat leftImg = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
    cv::Mat rightImg = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);

    if(leftImg.empty()) {
        std::cerr << "Error loading left image: " << leftPath << std::endl;
        return -1;
    }
    if(rightImg.empty()) {
        std::cerr << "Error loading right image: " << rightPath << std::endl;
        return -1;
    }

    // Ensure images have the same number of rows (epipolar lines)
    if(leftImg.rows != rightImg.rows) {
        std::cerr << "Left and right images must have the same number of rows (epipolar lines)." << std::endl;
        return -1;
    }

    // Load ground truth disparity map
    cv::Mat gtDisparity = cv::imread(gtPath, cv::IMREAD_UNCHANGED); // Use IMREAD_UNCHANGED to preserve precision
    if(gtDisparity.empty()) {
        std::cerr << "Error loading ground truth disparity map: " << gtPath << std::endl;
        return -1;
    }

    // Adjust for scaling factor
    double scalingFactor = 1.0 / 256.0; // Adjust based on dataset specifications
    cv::Mat gtDisparityFloat;
    gtDisparity.convertTo(gtDisparityFloat, CV_32F, scalingFactor);

    // Verify ground truth disparity statistics
    double minVal, maxVal;
    cv::minMaxLoc(gtDisparityFloat, &minVal, &maxVal);
    std::cout << "Ground Truth Disparity - Min: " << minVal << ", Max: " << maxVal << std::endl;

    int totalRows = leftImg.rows;
    int totalCols = leftImg.cols;

    // Initialize disparity map
    cv::Mat disparityMap(totalRows, totalCols, CV_32F, cv::Scalar(-1.0f));

    // Initialize StereoMatcher with adjusted scoring parameters
    int matchScore = 12;         // Increased reward for matches
    int mismatchPenalty = -3;    // Reduced penalty for mismatches
    int gapPenalty = -5;         // Reduced penalty for gaps
    StereoMatcher matcher(matchScore, mismatchPenalty, gapPenalty);

    // Start timing for NW method
    auto startNW = std::chrono::high_resolution_clock::now();

    // Parallel processing using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for(int row = 0; row < totalRows; ++row) {
        // Extract scanlines from images
        std::vector<int> leftLine(totalCols);
        std::vector<int> rightLine(totalCols);
        for(int col = 0; col < totalCols; ++col) {
            leftLine[col] = static_cast<int>(leftImg.at<uchar>(row, col));
        }
        for(int col = 0; col < totalCols; ++col) {
            rightLine[col] = static_cast<int>(rightImg.at<uchar>(row, col));
        }

        // Compute alignment based on selected strategy
        AlignmentResult result;
        if(strategy == "cpu") {
            result = matcher.computeAlignment(leftLine, rightLine);
        }
        else if(strategy == "cuda_wavefront") {
            result = matcher.computeAlignmentCUDA(leftLine, rightLine);
        }
        else if(strategy == "cuda_wavefront_fused") {
            result = matcher.computeAlignmentCUDA_Fused(leftLine, rightLine);
        }
        else {
            #pragma omp critical
            {
                std::cerr << "Unknown strategy: " << strategy << std::endl;
                std::cerr << "Available strategies: cpu, cuda_wavefront, cuda_wavefront_fused" << std::endl;
            }
            continue; // Skip this row
        }

        // Extract per-pixel disparities from the alignment result
        std::vector<float> estimatedDisparities(totalCols, -1.0f); // Initialize with -1

        size_t idxLeftPos = 0;   // Position in the original left sequence
        size_t idxRightPos = 0;  // Position in the original right sequence

        for(size_t idx = 0; idx < result.leftAligned.size(); ++idx) {
            if(result.leftAligned[idx] != -1 && result.rightAligned[idx] != -1) {
                // Assign disparity based on positional difference
                int disparity = static_cast<int>(idxRightPos) - static_cast<int>(idxLeftPos);
                if(idxLeftPos < estimatedDisparities.size()) {
                    estimatedDisparities[idxLeftPos] = static_cast<float>(disparity);
                }
                idxLeftPos++;
                idxRightPos++;
            }
            else {
                // Handle gaps by incrementing the respective indices
                if(result.leftAligned[idx] != -1) {
                    idxLeftPos++;
                }
                if(result.rightAligned[idx] != -1) {
                    idxRightPos++;
                }
            }
        }

        // Assign disparities to the disparity map
        for(int col = 0; col < totalCols; ++col) {
            disparityMap.at<float>(row, col) = estimatedDisparities[col];
        }
    }

    // Stop timing
    auto endNW = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> alignmentTime = endNW - startNW;
    std::cout << "Alignment Time (" << strategy << "): " << alignmentTime.count() << " ms" << std::endl;

    // Compute error metrics for the entire image
    double totalErrorNW = 0.0;
    int validPixelCountNW = 0;
    int badPixelCountNW = 0;
    double threshold = 3.0; // Example threshold for PBP

    for(int row = 0; row < totalRows; ++row) {
        for(int col = 0; col < totalCols; ++col) {
            float estDisp = disparityMap.at<float>(row, col);
            float gtDisp = gtDisparityFloat.at<float>(row, col);

            if(estDisp >= 0 && gtDisp > 0) { // Valid disparities
                float error = std::abs(estDisp - gtDisp);
                totalErrorNW += error;
                validPixelCountNW++;

                if(error > threshold) {
                    badPixelCountNW++;
                }
            }
        }
    }

    // Compute MAE and RMSE
    double MAE = (validPixelCountNW > 0) ? (totalErrorNW / validPixelCountNW) : 0.0;
    double RMSE = 0.0;
    if(validPixelCountNW > 0) {
        double totalSqError = 0.0;
        for(int row = 0; row < totalRows; ++row) {
            for(int col = 0; col < totalCols; ++col) {
                float estDisp = disparityMap.at<float>(row, col);
                float gtDisp = gtDisparityFloat.at<float>(row, col);
                if(estDisp >= 0 && gtDisp > 0) {
                    double error = estDisp - gtDisp;
                    totalSqError += error * error;
                }
            }
        }
        RMSE = std::sqrt(totalSqError / validPixelCountNW);
    }

    // Compute Percentage of Bad Pixels (PBP)
    double PBP = (validPixelCountNW > 0) ? ((double)badPixelCountNW / validPixelCountNW) * 100.0 : 0.0;

    // Compute mean estimated and ground truth disparities
    double sumEstDisparityNW = 0.0;
    int estValidCountNW = 0;
    double sumGtDisparity = 0.0;
    int gtValidCount = 0;

    for(int row = 0; row < totalRows; ++row) {
        for(int col = 0; col < totalCols; ++col) {
            float estDisp = disparityMap.at<float>(row, col);
            float gtDisp = gtDisparityFloat.at<float>(row, col);
            if(estDisp >= 0) {
                sumEstDisparityNW += estDisp;
                estValidCountNW++;
            }
            if(gtDisp > 0) {
                sumGtDisparity += gtDisp;
                gtValidCount++;
            }
        }
    }
    double meanEstDisparityNW = (estValidCountNW > 0) ? (sumEstDisparityNW / estValidCountNW) : 0.0;
    double meanGtDisparity = (gtValidCount > 0) ? (sumGtDisparity / gtValidCount) : 0.0;

    // Print the results for NW Method
    std::cout << "Mean Estimated Disparity (NW): " << meanEstDisparityNW << std::endl;
    std::cout << "Ground Truth Mean Disparity: " << meanGtDisparity << std::endl;
    std::cout << "Mean Absolute Error (MAE) (NW): " << MAE << std::endl;
    std::cout << "Root Mean Squared Error (RMSE) (NW): " << RMSE << std::endl;
    std::cout << "Percentage of Bad Pixels (PBP) (NW): " << PBP << "%" << std::endl;

    // Optional: Visualize the disparity map using the computed dynamic range
    cv::Mat validMask = (disparityMap >= 0); // Create a mask where valid pixels are true
    double computedMin, computedMax;
    cv::minMaxLoc(disparityMap, &computedMin, &computedMax, nullptr, nullptr, validMask);



    // Define ground truth range (adjust these values based on your dataset)
    float gtMin = 0.0f;
    float gtMax = 0.859375f;

    // Create a new matrix for the calibrated disparities
    cv::Mat calibratedDisp = disparityMap.clone();

    // Step 2: Remap each valid disparity value to the ground truth range
    for (int row = 0; row < disparityMap.rows; ++row) {
        for (int col = 0; col < disparityMap.cols; ++col) {
            float d = disparityMap.at<float>(row, col);
            if (d >= 0) { // Only process valid disparities
                float d_calibrated = ((d - static_cast<float>(computedMin)) / 
                                    (static_cast<float>(computedMax) - static_cast<float>(computedMin)))
                                   * (gtMax - gtMin) + gtMin;
                calibratedDisp.at<float>(row, col) = d_calibrated;
            }
        }
    }

    // Step 3: Convert calibrated disparity map to 8-bit for visualization.
    // Since calibrated values are in [gtMin, gtMax], scale by 255/(gtMax - gtMin)
    cv::Mat disparityMapVis;
    calibratedDisp.convertTo(disparityMapVis, CV_8U, 255.0 / (gtMax - gtMin));
    cv::applyColorMap(disparityMapVis, disparityMapVis, cv::COLORMAP_JET);

    // Save the resulting disparity image
    cv::imwrite("DisparityMap_NW.png", disparityMapVis);
    std::cout << "Disparity Map saved to: DisparityMap_NW.png" << std::endl;


    return 0;
}
