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
                ${CMAKE_SOURCE_DIR}/assets/Inquisition.tga
                ${CMAKE_CURRENT_BINARY_DIR}/Inquisition.tga)

add_custom_command(
        TARGET Exercise1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/assets/lizard.tga
                ${CMAKE_CURRENT_BINARY_DIR}/lizard.tga)

add_custom_command(
        TARGET Exercise1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/src/kernels/rotate.cl
                ${CMAKE_CURRENT_BINARY_DIR}/rotate.cl)


set(SOURCES_EXERCISE2
	src/opencl_manager.cpp
	src/exercise2.cpp
	src/prefix_sum.cpp
)

add_executable(Exercise2 ${SOURCES_EXERCISE2})
target_include_directories(Exercise2 PRIVATE ${INCLUDE})
target_compile_features(Exercise2 PRIVATE cxx_std_17)
target_link_libraries(Exercise2 OpenCL::OpenCL)

add_custom_command(
        TARGET Exercise2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/src/kernels/scan.cl
                ${CMAKE_CURRENT_BINARY_DIR}/scan.cl)

set(SOURCES_EXERCISE3
	src/opencl_manager.cpp
	src/exercise3.cpp
	src/prefix_sum.cpp
	src/stream_compact.cpp
)

add_executable(Exercise3 ${SOURCES_EXERCISE3})
target_include_directories(Exercise3 PRIVATE ${INCLUDE})
target_compile_features(Exercise3 PRIVATE cxx_std_17)
target_link_libraries(Exercise3 OpenCL::OpenCL)

add_custom_command(
        TARGET Exercise3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/src/kernels/stream_compact.cl
                ${CMAKE_CURRENT_BINARY_DIR}/stream_compact.cl)
