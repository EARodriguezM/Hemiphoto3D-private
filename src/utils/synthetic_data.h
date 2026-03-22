#pragma once

#include <string>

// Generate a synthetic test dataset: textured sphere rendered from 36 viewpoints.
// Creates JPEG images, ground-truth camera poses (JSON), and ground-truth mesh (OBJ).
// Returns true on success.
bool generateSyntheticData(const std::string& output_dir);
