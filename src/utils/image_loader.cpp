#include "utils/image_loader.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// ============================================================================
// Sensor width lookup table (mm) — common smartphones
// ============================================================================

static double lookupSensorWidth(const std::string& make, const std::string& model) {
    // Normalize to lowercase for matching
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return s;
    };

    std::string lmake = lower(make);
    std::string lmodel = lower(model);

    // Apple iPhones: main camera sensor widths
    if (lmake.find("apple") != std::string::npos) {
        // iPhone 15 Pro Max: 1/1.28" ≈ 9.8mm, iPhone 15 Pro: 1/1.28" ≈ 9.8mm
        if (lmodel.find("15 pro") != std::string::npos) return 9.8;
        // iPhone 14 Pro/15/14: 1/1.65" ≈ 7.6mm
        if (lmodel.find("14 pro") != std::string::npos) return 7.6;
        // iPhone 13/14/12 Pro: ~6.17mm
        return 6.17;
    }

    // Google Pixel
    if (lmake.find("google") != std::string::npos) {
        if (lmodel.find("pixel 8 pro") != std::string::npos) return 8.2;
        if (lmodel.find("pixel 8") != std::string::npos) return 6.17;
        if (lmodel.find("pixel 7 pro") != std::string::npos) return 8.2;
        return 6.17;
    }

    // Samsung Galaxy
    if (lmake.find("samsung") != std::string::npos) {
        if (lmodel.find("s24 ultra") != std::string::npos) return 8.6;
        if (lmodel.find("s24") != std::string::npos) return 6.4;
        if (lmodel.find("s23 ultra") != std::string::npos) return 8.6;
        if (lmodel.find("s23") != std::string::npos) return 6.4;
        return 6.4;
    }

    return 0.0;  // unknown
}

// ============================================================================
// Minimal EXIF parser — reads APP1 segment from JPEG
// ============================================================================

// Read a 16-bit big-endian or little-endian value
static uint16_t read16(const uint8_t* p, bool big_endian) {
    if (big_endian) return (uint16_t(p[0]) << 8) | p[1];
    return (uint16_t(p[1]) << 8) | p[0];
}

static uint32_t read32(const uint8_t* p, bool big_endian) {
    if (big_endian)
        return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
               (uint32_t(p[2]) << 8) | p[3];
    return (uint32_t(p[3]) << 24) | (uint32_t(p[2]) << 16) |
           (uint32_t(p[1]) << 8) | p[0];
}

static double readRational(const uint8_t* data, uint32_t offset, bool big_endian) {
    uint32_t num = read32(data + offset, big_endian);
    uint32_t den = read32(data + offset + 4, big_endian);
    if (den == 0) return 0.0;
    return double(num) / double(den);
}

static std::string readString(const uint8_t* data, uint32_t offset, uint32_t count) {
    std::string s(reinterpret_cast<const char*>(data + offset), count);
    // Trim trailing nulls and whitespace
    while (!s.empty() && (s.back() == '\0' || s.back() == ' '))
        s.pop_back();
    return s;
}

EXIFData extractEXIF(const std::string& filepath) {
    EXIFData exif;

    // Only parse JPEG files
    std::string ext = fs::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext != ".jpg" && ext != ".jpeg") return exif;

    std::ifstream file(filepath, std::ios::binary);
    if (!file) return exif;

    // Read up to 64KB (EXIF is in the first APP1 marker)
    std::vector<uint8_t> buf(65536);
    file.read(reinterpret_cast<char*>(buf.data()), buf.size());
    size_t bytes_read = file.gcount();
    if (bytes_read < 12) return exif;

    // Check JPEG SOI
    if (buf[0] != 0xFF || buf[1] != 0xD8) return exif;

    // Find APP1 marker (0xFFE1)
    size_t pos = 2;
    while (pos + 4 < bytes_read) {
        if (buf[pos] != 0xFF) break;
        uint8_t marker = buf[pos + 1];
        uint16_t seg_len = (uint16_t(buf[pos + 2]) << 8) | buf[pos + 3];

        if (marker == 0xE1) {
            // APP1 found — check for "Exif\0\0" header
            if (pos + 10 < bytes_read &&
                buf[pos + 4] == 'E' && buf[pos + 5] == 'x' &&
                buf[pos + 6] == 'i' && buf[pos + 7] == 'f' &&
                buf[pos + 8] == 0 && buf[pos + 9] == 0) {

                // TIFF header starts at pos+10
                const uint8_t* tiff = &buf[pos + 10];
                size_t tiff_len = seg_len - 8;  // segment length minus "Exif\0\0"

                bool big_endian = (tiff[0] == 'M' && tiff[1] == 'M');
                // little_endian: 'I','I'

                uint32_t ifd_offset = read32(tiff + 4, big_endian);
                if (ifd_offset >= tiff_len) return exif;

                // Parse IFD0
                uint16_t num_entries = read16(tiff + ifd_offset, big_endian);

                for (int i = 0; i < num_entries; i++) {
                    uint32_t entry_offset = ifd_offset + 2 + i * 12;
                    if (entry_offset + 12 > tiff_len) break;

                    uint16_t tag = read16(tiff + entry_offset, big_endian);
                    uint16_t type = read16(tiff + entry_offset + 2, big_endian);
                    uint32_t count = read32(tiff + entry_offset + 4, big_endian);
                    uint32_t value_offset = read32(tiff + entry_offset + 8, big_endian);

                    // Tag 0x0112 = Orientation
                    if (tag == 0x0112 && type == 3) {  // SHORT
                        exif.orientation = read16(tiff + entry_offset + 8, big_endian);
                    }
                    // Tag 0x010F = Make
                    else if (tag == 0x010F && type == 2) {  // ASCII
                        if (count <= 4) {
                            exif.camera_make = readString(tiff + entry_offset + 8, 0, count);
                        } else if (value_offset + count <= tiff_len) {
                            exif.camera_make = readString(tiff, value_offset, count);
                        }
                    }
                    // Tag 0x0110 = Model
                    else if (tag == 0x0110 && type == 2) {
                        if (count <= 4) {
                            exif.camera_model = readString(tiff + entry_offset + 8, 0, count);
                        } else if (value_offset + count <= tiff_len) {
                            exif.camera_model = readString(tiff, value_offset, count);
                        }
                    }
                    // Tag 0x8769 = ExifIFD pointer
                    else if (tag == 0x8769 && type == 4) {  // LONG
                        // Follow pointer to Exif sub-IFD for focal length
                        if (value_offset < tiff_len) {
                            uint16_t sub_entries = read16(tiff + value_offset, big_endian);
                            for (int j = 0; j < sub_entries; j++) {
                                uint32_t sub_off = value_offset + 2 + j * 12;
                                if (sub_off + 12 > tiff_len) break;

                                uint16_t sub_tag = read16(tiff + sub_off, big_endian);
                                uint16_t sub_type = read16(tiff + sub_off + 2, big_endian);
                                uint32_t sub_value_off = read32(tiff + sub_off + 8, big_endian);

                                // Tag 0x920A = FocalLength (RATIONAL)
                                if (sub_tag == 0x920A && sub_type == 5) {
                                    if (sub_value_off + 8 <= tiff_len) {
                                        exif.focal_length_mm = readRational(tiff, sub_value_off, big_endian);
                                    }
                                }
                            }
                        }
                    }
                }

                exif.valid = true;
                return exif;
            }
        }

        // Skip to next marker
        pos += 2 + seg_len;
    }

    return exif;
}

// ============================================================================
// Sensor width estimation
// ============================================================================

double estimateSensorWidth(const std::string& make, const std::string& model) {
    double w = lookupSensorWidth(make, model);
    if (w > 0.0) return w;
    return 6.0;  // reasonable default for most smartphones
}

// ============================================================================
// EXIF orientation handling
// ============================================================================

void applyEXIFOrientation(cv::Mat& image, int orientation) {
    switch (orientation) {
        case 1: break;  // normal
        case 2: cv::flip(image, image, 1); break;  // horizontal flip
        case 3: cv::rotate(image, image, cv::ROTATE_180); break;
        case 4: cv::flip(image, image, 0); break;  // vertical flip
        case 5:
            cv::transpose(image, image);
            break;
        case 6:
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
            break;
        case 7:
            cv::transpose(image, image);
            cv::flip(image, image, 1);
            break;
        case 8:
            cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default: break;
    }
}

// ============================================================================
// Image loading
// ============================================================================

static bool isImageFile(const std::string& ext) {
    std::string e = ext;
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".tiff" || e == ".tif";
}

std::vector<ImageData> loadImages(const std::string& dir, const PipelineConfig& config) {
    std::vector<ImageData> images;

    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        fprintf(stderr, "Error: Input directory '%s' does not exist.\n", dir.c_str());
        return images;
    }

    // Collect image paths
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && isImageFile(entry.path().extension().string())) {
            paths.push_back(entry.path());
        }
    }

    // Sort by filename for deterministic ordering
    std::sort(paths.begin(), paths.end(),
              [](const fs::path& a, const fs::path& b) {
                  return a.filename().string() < b.filename().string();
              });

    if (paths.empty()) {
        fprintf(stderr, "Error: No image files found in '%s'.\n", dir.c_str());
        return images;
    }

    if (config.verbose) {
        printf("Found %zu image files in %s\n", paths.size(), dir.c_str());
    }

    if (paths.size() < 10) {
        fprintf(stderr, "Warning: Only %zu images found. Recommend at least 30 for good reconstruction.\n",
                paths.size());
    } else if (paths.size() < 30) {
        fprintf(stderr, "Note: %zu images found. 30+ images recommended for best results.\n",
                paths.size());
    }

    for (size_t i = 0; i < paths.size(); i++) {
        const auto& path = paths[i];
        ImageData img;
        img.filename = path.filename().string();
        img.id = static_cast<int>(i);

        // Load BGR image
        img.image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (img.image.empty()) {
            fprintf(stderr, "Warning: Failed to load '%s', skipping.\n", path.string().c_str());
            continue;
        }

        // Extract EXIF before any transforms
        EXIFData exif = extractEXIF(path.string());

        // Apply EXIF orientation
        if (exif.valid && exif.orientation != 1) {
            applyEXIFOrientation(img.image, exif.orientation);
        }

        // Resize if needed
        int max_dim = std::max(img.image.cols, img.image.rows);
        if (max_dim > config.max_image_size) {
            double scale = double(config.max_image_size) / max_dim;
            cv::resize(img.image, img.image, cv::Size(), scale, scale, cv::INTER_AREA);
        }

        img.width = img.image.cols;
        img.height = img.image.rows;

        // Convert to grayscale float32 [0,1]
        cv::Mat gray_u8;
        cv::cvtColor(img.image, gray_u8, cv::COLOR_BGR2GRAY);
        gray_u8.convertTo(img.gray, CV_32F, 1.0 / 255.0);

        // Compute intrinsics
        double focal_mm = config.focal_length_mm;
        double sensor_mm = config.sensor_width_mm;

        if (focal_mm <= 0.0 && exif.valid && exif.focal_length_mm > 0.0) {
            focal_mm = exif.focal_length_mm;
        }
        if (sensor_mm <= 0.0) {
            if (exif.valid && !exif.camera_make.empty()) {
                sensor_mm = estimateSensorWidth(exif.camera_make, exif.camera_model);
            } else {
                sensor_mm = 6.0;  // default
            }
        }

        if (focal_mm > 0.0) {
            img.focal_length_px = focal_mm * img.width / sensor_mm;
        } else {
            // Fallback: assume 28mm equivalent focal length
            // 28mm equiv → physical ≈ 28 * sensor_mm / 36
            double focal_equiv_mm = 28.0 * sensor_mm / 36.0;
            img.focal_length_px = focal_equiv_mm * img.width / sensor_mm;
        }

        double fx = img.focal_length_px;
        double cx = img.width / 2.0;
        double cy = img.height / 2.0;

        img.K = (cv::Mat_<double>(3, 3) <<
                 fx,  0, cx,
                  0, fx, cy,
                  0,  0,  1);

        img.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

        if (config.verbose) {
            printf("  [%02d] %-30s  %dx%d  f=%.1fpx",
                   img.id, img.filename.c_str(), img.width, img.height, fx);
            if (exif.valid && exif.focal_length_mm > 0.0) {
                printf("  (EXIF: %.1fmm, %s %s)",
                       exif.focal_length_mm, exif.camera_make.c_str(), exif.camera_model.c_str());
            }
            printf("\n");
        }

        images.push_back(std::move(img));
    }

    if (config.verbose) {
        printf("Loaded %zu images successfully.\n", images.size());
    }

    return images;
}

// Convenience overload
std::vector<ImageData> loadImages(const std::string& dir, int max_size) {
    PipelineConfig config;
    config.max_image_size = max_size;
    return loadImages(dir, config);
}
