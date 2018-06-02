#pragma once
#include <tga.h>

namespace cl {
    class Context;
    class CommandQueue;
    class Kernel;
}

tga::TGAImage load_tga_image(const std::string& file);
void rotate_image(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, tga::TGAImage& image, float theta);
