cmake_minimum_required(VERSION 3.13)
project(example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wconversion")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for nvim LSP support

find_package(Eigen3 REQUIRED)

add_executable(example
    example.cpp
    rigid_transform_3D.hpp
)

target_link_libraries(example Eigen3::Eigen)
