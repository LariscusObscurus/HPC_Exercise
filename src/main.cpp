#include "CL/cl.hpp"
#include <string>
#include <tga.h>
#include <functional>
#include <opencl_manager.h>
#include <rotate_image.h>
#include <constants/rotation_constants.h>

#define __CL_ENABLE_EXCEPTIONS



int main(int argc, char* argv[])
{

    auto open_cl = opencl_manager{};
    open_cl.load_kernel(kernel_file, "rotate_image");

    auto&& tga_image = load_tga_image(image_file);

    std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel kernel, tga::TGAImage& image, float theta)> fun = rotate_image;
    open_cl.execute_kernel<tga::TGAImage&, const float>(fun, tga_image, std::move(theta));

    std::getchar();
    return 0;
}