#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "types.h"
#include "pipeline.h"

static void printUsage(const char* progname) {
    printf("Usage: %s [options] -i <input_dir> -o <output_path>\n", progname);
    printf("\nCUDA-accelerated 3D reconstruction from smartphone images.\n");
    printf("\nRequired:\n");
    printf("  -i, --input <dir>              Input image directory\n");
    printf("  -o, --output <path>            Output mesh (.obj, .stl, .ply)\n");
    printf("\nQuality:\n");
    printf("  --quality <preset>             low, medium, high, ultra (default: high)\n");
    printf("\n  Presets:\n");
    printf("    low:    max_image=1600, poisson_depth=8,  mvs_res=0.50, mvs_iter=4\n");
    printf("    medium: max_image=2400, poisson_depth=9,  mvs_res=0.75, mvs_iter=6\n");
    printf("    high:   max_image=3200, poisson_depth=10, mvs_res=1.00, mvs_iter=12\n");
    printf("    ultra:  max_image=4800, poisson_depth=12, mvs_res=1.00, mvs_iter=12\n");
    printf("\nFull options (override presets):\n");
    printf("  --max-image-size <px>          Max image dimension (default: from preset)\n");
    printf("  --match-ratio <float>          Lowe's ratio threshold (default: 0.75)\n");
    printf("  --mvs-resolution <float>       MVS resolution multiplier (default: from preset)\n");
    printf("  --mvs-iterations <int>         PatchMatch iterations (default: from preset)\n");
    printf("  --poisson-depth <int>          Octree depth (default: from preset)\n");
    printf("  --smooth-iterations <int>      Mesh smoothing passes (default: 3)\n");
    printf("  --decimate <int>               Target face count, 0=none (default: 0)\n");
    printf("  --export-pointcloud <path>     Also save dense cloud as .ply\n");
    printf("\nCamera:\n");
    printf("  --focal-length <mm>            Override focal length\n");
    printf("  --sensor-width <mm>            Override sensor width\n");
    printf("  --turntable                    Assume turntable capture\n");
    printf("  --scale-bar <mm>               Known scale bar length for calibration\n");
    printf("\nGPU:\n");
    printf("  --gpu <id>                     Device ID (default: 0)\n");
    printf("  --gpu-memory <MB>              Max GPU memory (default: auto)\n");
    printf("\nOutput:\n");
    printf("  --verbose                      Detailed progress\n");
    printf("  --save-intermediate            Save depth maps, sparse cloud, etc.\n");
    printf("  --log <path>                   Log to file\n");
    printf("  --resume                       Skip stages whose outputs already exist\n");
    printf("  -h, --help                     Show this help message\n");
    printf("\nExample:\n");
    printf("  %s -i photos/ -o model.obj --quality high --verbose\n", progname);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    PipelineConfig config;

    // Track which preset-overridable fields were explicitly set by the user,
    // so we can apply the preset first then re-apply explicit overrides.
    struct {
        int max_image_size = -1;
        float mvs_resolution = -1;
        int mvs_iterations = -1;
        int poisson_depth = -1;
        int smooth_iterations = -1;
    } overrides;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        // Required
        else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            config.input_dir = argv[++i];
        }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.output_path = argv[++i];
        }
        // Quality preset
        else if (arg == "--quality" && i + 1 < argc) {
            config.quality = argv[++i];
        }
        // Overridable preset values
        else if (arg == "--max-image-size" && i + 1 < argc) {
            overrides.max_image_size = std::atoi(argv[++i]);
        }
        else if (arg == "--mvs-resolution" && i + 1 < argc) {
            overrides.mvs_resolution = (float)std::atof(argv[++i]);
        }
        else if (arg == "--mvs-iterations" && i + 1 < argc) {
            overrides.mvs_iterations = std::atoi(argv[++i]);
        }
        else if (arg == "--poisson-depth" && i + 1 < argc) {
            overrides.poisson_depth = std::atoi(argv[++i]);
        }
        else if (arg == "--smooth-iterations" && i + 1 < argc) {
            overrides.smooth_iterations = std::atoi(argv[++i]);
        }
        // Non-preset options
        else if (arg == "--match-ratio" && i + 1 < argc) {
            config.match_ratio = (float)std::atof(argv[++i]);
        }
        else if (arg == "--decimate" && i + 1 < argc) {
            config.decimate_target = std::atoi(argv[++i]);
        }
        else if ((arg == "--export-pointcloud" || arg == "--pointcloud") && i + 1 < argc) {
            config.pointcloud_path = argv[++i];
        }
        // Camera
        else if (arg == "--focal-length" && i + 1 < argc) {
            config.focal_length_mm = std::atof(argv[++i]);
        }
        else if (arg == "--sensor-width" && i + 1 < argc) {
            config.sensor_width_mm = std::atof(argv[++i]);
        }
        else if (arg == "--turntable") {
            config.turntable = true;
        }
        else if (arg == "--scale-bar" && i + 1 < argc) {
            config.scale_bar_length_mm = std::atof(argv[++i]);
        }
        // GPU
        else if (arg == "--gpu" && i + 1 < argc) {
            config.gpu_id = std::atoi(argv[++i]);
        }
        else if (arg == "--gpu-memory" && i + 1 < argc) {
            config.gpu_memory_limit = (size_t)std::atol(argv[++i]) * 1024 * 1024;
        }
        // Output control
        else if (arg == "--verbose") {
            config.verbose = true;
        }
        else if (arg == "--save-intermediate") {
            config.save_intermediate = true;
        }
        else if (arg == "--log" && i + 1 < argc) {
            config.log_path = argv[++i];
        }
        else if (arg == "--resume") {
            config.resume = true;
            config.save_intermediate = true;  // resume implies save
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate required args
    if (config.input_dir.empty() || config.output_path.empty()) {
        fprintf(stderr, "Error: --input and --output are required.\n");
        printUsage(argv[0]);
        return 1;
    }

    // Validate quality preset
    if (config.quality != "low" && config.quality != "medium" &&
        config.quality != "high" && config.quality != "ultra") {
        fprintf(stderr, "Error: invalid quality preset '%s'. Use low, medium, high, or ultra.\n",
                config.quality.c_str());
        return 1;
    }

    // Apply quality preset, then re-apply any explicit user overrides
    config.applyPreset();
    if (overrides.max_image_size >= 0)   config.max_image_size   = overrides.max_image_size;
    if (overrides.mvs_resolution >= 0)   config.mvs_resolution   = overrides.mvs_resolution;
    if (overrides.mvs_iterations >= 0)   config.mvs_iterations   = overrides.mvs_iterations;
    if (overrides.poisson_depth >= 0)    config.poisson_depth    = overrides.poisson_depth;
    if (overrides.smooth_iterations >= 0) config.smooth_iterations = overrides.smooth_iterations;

    // Run pipeline
    Pipeline pipeline(config);
    bool success = pipeline.run();

    return success ? 0 : 1;
}
