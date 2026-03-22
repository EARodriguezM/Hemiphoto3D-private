cmake_minimum_required(VERSION 3.24)
project(cuda_3d_recon LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures: supported by CUDA 13.0 (minimum compute_75)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# CUDA compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=esa_on_defaulted_function_ignored\"")

# Find dependencies
include(cmake/FindDependencies.cmake)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Collect source files
set(RECON_SOURCES
    src/main.cpp
    src/pipeline.cu
    src/utils/cuda_utils.cu
    src/utils/image_loader.cpp
    src/utils/synthetic_data.cpp
    src/feature_detection/sift_cuda.cu
    src/feature_detection/feature_matching.cu
    src/sfm/camera_pose.cpp
    src/sfm/bundle_adjust.cpp
    src/dense_recon/patchmatch_mvs.cu
    src/dense_recon/point_cloud.cu
    src/meshing/poisson_recon.cpp
    src/meshing/marching_cubes.cu
    src/export/obj_exporter.cpp
    src/export/stl_exporter.cpp
    src/export/ply_exporter.cpp
)

# Main executable
add_executable(recon3d ${RECON_SOURCES})
target_link_libraries(recon3d
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    ${CUBLAS_LIB}
    ${CUSPARSE_LIB}
    ${CURAND_LIB}
)

# Enable testing
enable_testing()
add_subdirectory(tests)
