cmake_minimum_required(VERSION 3.9)
project(Exercise1)

find_package(OpenCL REQUIRED)

add_executable(Exercise1 src/main.cpp)
target_include_directories(Exercise1 PRIVATE src)
target_compile_features(Exercise1 PRIVATE cxx_std_17)
target_link_libraries(Exercise1 OpenCL::OpenCL)