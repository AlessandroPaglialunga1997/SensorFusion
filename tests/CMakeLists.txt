enable_testing()

add_executable(test_ukf_imu_gps test_ukf_imu_gps.cpp)
target_link_libraries(test_ukf_imu_gps PRIVATE GTest::gtest_main ukf_manifold_library)

include(GoogleTest)
gtest_discover_tests(test_ukf_imu_gps)
