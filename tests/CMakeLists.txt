# Test executable
set(TEST_SOURCES
    test_image_loader.cpp
    test_sift.cpp
    test_matching.cpp
    test_sfm.cpp
    test_meshing.cpp
)

# Sources needed by tests
set(TEST_LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/image_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/synthetic_data.cpp
    ${CMAKE_SOURCE_DIR}/src/feature_detection/sift_cuda.cu
)

add_executable(recon3d_tests ${TEST_SOURCES} ${TEST_LIB_SOURCES})
target_link_libraries(recon3d_tests
    GTest::GTest
    GTest::Main
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(recon3d_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
