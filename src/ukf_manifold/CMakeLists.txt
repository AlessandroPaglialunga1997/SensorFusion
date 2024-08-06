add_library(ukf_manifold_library SHARED
  ukf.hpp 
  nav_state.cpp
  odometry_state.cpp)

target_compile_features(ukf_manifold_library PUBLIC)
