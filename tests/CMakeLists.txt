# Test executable
set(TEST_SOURCES
    test_image_loader.cpp
    test_sift.cpp
    test_matching.cpp
    test_sfm.cpp
    test_meshing.cpp
)

# Sources needed by tests (includes CUDA files)
set(TEST_LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/synthetic_data.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/cuda_utils.cu
    ${CMAKE_SOURCE_DIR}/src/feature_detection/sift_cuda.cu
)

add_executable(recon3d_tests ${TEST_SOURCES} ${TEST_LIB_SOURCES})
target_link_libraries(recon3d_tests
    GTest::GTest
    GTest::Main
    cudasift_lib
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Ensure CUDA runtime is linked
find_library(CUDART_LIB cudart HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
if(CUDART_LIB)
    target_link_libraries(recon3d_tests ${CUDART_LIB})
endif()

include(GoogleTest)
gtest_discover_tests(recon3d_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
