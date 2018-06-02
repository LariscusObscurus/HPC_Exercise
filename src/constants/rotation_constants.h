#pragma once
#include <string>

#define M_PI       3.14159265358979323846   // pi

const std::string kernel_file = "rotate.cl";
const std::string image_file = "lenna.tga";
const std::string output_image_file = "rotated_lenna.tga";

const float theta = M_PI / 4;
