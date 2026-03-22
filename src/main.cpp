#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "types.h"
#include "pipeline.h"
#include "utils/image_loader.h"

static void printUsage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("\nCUDA-accelerated 3D reconstruction from smartphone images.\n");
    printf("\nRequired:\n");
    printf("  -i, --input <dir>        Input image directory\n");
    printf("  -o, --output <file>      Output mesh file (.obj, .stl, or .ply)\n");
    printf("\nOptional:\n");
    printf("  --quality <preset>       Quality preset: low, medium, high, ultra (default: high)\n");
    printf("  --max-image-size <px>    Max image dimension (default: 3200)\n");
    printf("  --focal-length <mm>      Override focal length (default: from EXIF)\n");
    printf("  --sensor-width <mm>      Override sensor width (default: estimate)\n");
    printf("  --turntable              Assume turntable capture pattern\n");
    printf("  --save-intermediate      Save intermediate results for checkpoint/resume\n");
    printf("  --pointcloud <file>      Export point cloud as .ply\n");
    printf("  --gpu <id>               GPU device ID (default: 0)\n");
    printf("  --verbose                Verbose logging\n");
    printf("  --scale-bar <mm>         Known scale bar length for calibration\n");
    printf("  --poisson-depth <n>      Poisson reconstruction depth (default: 10)\n");
    printf("  --mvs-resolution <f>     MVS downscale factor (default: 1.0)\n");
    printf("  --decimate <n>           Target face count for decimation (0 = none)\n");
    printf("  -h, --help               Show this help message\n");
    printf("\nExample:\n");
    printf("  %s -i photos/ -o model.obj --quality high --verbose\n", progname);
}

static PipelineConfig parseArgs(int argc, char** argv) {
    PipelineConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if ((arg == "-i" || arg == "--input") && i+1 < argc) {
            config.input_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i+1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--quality" && i+1 < argc) {
            config.quality = argv[++i];
        } else if (arg == "--max-image-size" && i+1 < argc) {
            config.max_image_size = std::atoi(argv[++i]);
        } else if (arg == "--focal-length" && i+1 < argc) {
            config.focal_length_mm = std::atof(argv[++i]);
        } else if (arg == "--sensor-width" && i+1 < argc) {
            config.sensor_width_mm = std::atof(argv[++i]);
        } else if (arg == "--turntable") {
            config.turntable = true;
        } else if (arg == "--save-intermediate") {
            config.save_intermediate = true;
        } else if (arg == "--pointcloud" && i+1 < argc) {
            config.pointcloud_path = argv[++i];
        } else if (arg == "--gpu" && i+1 < argc) {
            config.gpu_id = std::atoi(argv[++i]);
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--scale-bar" && i+1 < argc) {
            config.scale_bar_length_mm = std::atof(argv[++i]);
        } else if (arg == "--poisson-depth" && i+1 < argc) {
            config.poisson_depth = std::atoi(argv[++i]);
        } else if (arg == "--mvs-resolution" && i+1 < argc) {
            config.mvs_resolution = std::atof(argv[++i]);
        } else if (arg == "--decimate" && i+1 < argc) {
            config.decimate_target = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            printUsage(argv[0]);
            exit(1);
        }
    }

    return config;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    PipelineConfig config = parseArgs(argc, argv);

    if (config.input_dir.empty() || config.output_path.empty()) {
        fprintf(stderr, "Error: --input and --output are required.\n");
        printUsage(argv[0]);
        return 1;
    }

    config.applyPreset();

    // Load images
    auto images = loadImages(config.input_dir, config);
    if (images.empty()) {
        fprintf(stderr, "Error: No images loaded from '%s'.\n", config.input_dir.c_str());
        return 1;
    }

    printf("Loaded %zu images.\n", images.size());

    // TODO: Invoke pipeline
    // Pipeline pipeline(config);
    // pipeline.run();

    printf("Pipeline not yet implemented. Image loading successful.\n");
    return 0;
}
