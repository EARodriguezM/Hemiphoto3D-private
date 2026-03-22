#pragma once

#include "types.h"
#include <vector>
#include <string>

// Load all images from a directory, extracting EXIF and computing intrinsics.
// Uses config overrides for focal_length_mm and sensor_width_mm if non-zero.
std::vector<ImageData> loadImages(const std::string& dir, const PipelineConfig& config);

// Convenience overload using default config (max_size only)
std::vector<ImageData> loadImages(const std::string& dir, int max_size = 3200);

// EXIF data extracted from a JPEG file
struct EXIFData {
    double focal_length_mm = 0.0;
    std::string camera_make;
    std::string camera_model;
    int orientation = 1;  // EXIF orientation tag (1-8)
    bool valid = false;
};

// Extract EXIF data from a JPEG file (minimal parser for APP1 segment)
EXIFData extractEXIF(const std::string& filepath);

// Estimate sensor width (mm) from camera make/model
double estimateSensorWidth(const std::string& make, const std::string& model);

// Apply EXIF orientation to an image (rotates/flips in place)
void applyEXIFOrientation(cv::Mat& image, int orientation);
