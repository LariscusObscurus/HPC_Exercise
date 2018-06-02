cmake_minimum_required(VERSION 3.9)
project(Exercise1)

find_package(OpenCL REQUIRED)

set(SOURCES_EXERCISE1
	tga/tga.cpp
	src/rotate_image.cpp
	src/opencl_manager.cpp
	src/exercise1.cpp
)

set(INCLUDE
	tga
	src
)


add_executable(Exercise1 ${SOURCES_EXERCISE1})
target_include_directories(Exercise1 PRIVATE ${INCLUDE})
target_compile_features(Exercise1 PRIVATE cxx_std_17)
target_link_libraries(Exercise1 OpenCL::OpenCL)

add_custom_command(
        TARGET Exercise1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/assets/lenna.tga
                ${CMAKE_CURRENT_BINARY_DIR}/lenna.tga)

add_custom_command(
        TARGET Exercise1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/src/kernels/rotate.cl
                ${CMAKE_CURRENT_BINARY_DIR}/rotate.cl)


set(SOURCES_EXERCISE2
	src/opencl_manager.cpp
	src/exercise2.cpp
)

add_executable(Exercise2 ${SOURCES_EXERCISE2})
target_include_directories(Exercise2 PRIVATE ${INCLUDE})
target_compile_features(Exercise2 PRIVATE cxx_std_17)
target_link_libraries(Exercise2 OpenCL::OpenCL)

add_custom_command(
        TARGET Exercise1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/src/kernels/scan.cl
                ${CMAKE_CURRENT_BINARY_DIR}/scan.cl)
