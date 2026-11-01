# The Gazebo replay tool for `strathe fit-sim --sim gazebo` (PROJECT_STATUS.md #82).
#   cmake -S calibration/simulators/gazebo -B calibration/simulators/gazebo/build && cmake --build calibration/simulators/gazebo/build
# (or: python strathe.py setup-sim gazebo). Needs Gazebo Harmonic (gz-sim 8) installed: calibration/simulators/gazebo/README.md
cmake_minimum_required(VERSION 3.16)
project(strathe_gz_replay CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(gz-sim8 REQUIRED)
find_package(sdformat14 REQUIRED)
add_executable(strathe_gz_replay strathe_gz_replay.cpp)
target_link_libraries(strathe_gz_replay gz-sim8::gz-sim8 sdformat14::sdformat14)
