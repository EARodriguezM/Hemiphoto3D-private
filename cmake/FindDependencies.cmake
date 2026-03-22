# FindDependencies.cmake — Locate all required libraries

# OpenCV (with CUDA modules)
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc imgcodecs highgui calib3d features2d
    cudaarithm cudafeatures2d cudaimgproc cudawarping
)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION_STRING}")

# Ceres Solver
find_package(Ceres 2.1 REQUIRED)
message(STATUS "Ceres version: ${Ceres_VERSION}")

# Google Test
find_package(GTest REQUIRED)
message(STATUS "GTest found")

# CUDA libraries (cuBLAS, cuSPARSE, cuRAND)
find_library(CUBLAS_LIB cublas HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
find_library(CUSPARSE_LIB cusparse HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
find_library(CURAND_LIB curand HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
message(STATUS "cuBLAS: ${CUBLAS_LIB}")
message(STATUS "cuSPARSE: ${CUSPARSE_LIB}")
message(STATUS "cuRAND: ${CURAND_LIB}")
