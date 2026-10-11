# The replay tool `strathe fit-stonefish` drives (see README.md). Build it against your Stonefish install:
#   cmake -S calibration/stonefish -B calibration/stonefish/build -DCMAKE_PREFIX_PATH=<stonefish install prefix>
#   cmake --build calibration/stonefish/build
# Fitting added mass needs Stonefish with stonefish_added_mass.patch applied; without it the tool still runs and
# `fit-stonefish` fits drag, buoyancy and the centre of gravity only.
cmake_minimum_required(VERSION 3.16)
project(strathe_stonefish_replay CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Stonefish REQUIRED)
add_executable(strathe_stonefish_replay strathe_stonefish_replay.cpp)
target_link_libraries(strathe_stonefish_replay Stonefish::Stonefish)
