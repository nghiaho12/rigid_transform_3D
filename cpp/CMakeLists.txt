cmake_minimum_required(VERSION 3.16)
project(example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wconversion -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for nvim LSP support

find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)

add_library(rigid_transform
    src/rigid_transform.cpp
    src/rigid_transform.hpp
)

add_library(helper
    src/helper.cpp
    src/helper.hpp
)

add_executable(example
    src/example.cpp
    src/helper.hpp
    src/rigid_transform.hpp
)

add_executable(unit_tests
    src/unit_tests.cpp
    src/helper.hpp
    src/rigid_transform.hpp
)

target_link_libraries(example PRIVATE 
    Eigen3::Eigen 
    rigid_transform
    helper)

target_link_libraries(unit_tests PRIVATE 
    Eigen3::Eigen 
    Catch2::Catch2WithMain 
    rigid_transform
    helper)

include(CTest)
add_test(NAME unit_tests COMMAND unit_tests)
