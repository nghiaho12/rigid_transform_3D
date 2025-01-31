cmake_minimum_required(VERSION 3.16)
project(example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wconversion")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for nvim LSP support

find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)

add_executable(example
    example.cpp
    helper.hpp
    rigid_transform_3D.hpp
)

add_executable(unit_tests
    unit_tests.cpp
    helper.hpp
    rigid_transform_3D.hpp
)

target_link_libraries(example PRIVATE Eigen3::Eigen)
target_link_libraries(unit_tests PRIVATE Eigen3::Eigen Catch2::Catch2WithMain)

include(CTest)
add_test(NAME unit_tests COMMAND unit_tests)
