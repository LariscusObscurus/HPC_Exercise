#include <chrono>
#include <iostream>
#include <tga.h>
#include <functional>
#include <opencl_manager.h>
#include <rotate_image.h>
#include <constants/rotation_constants.h>

int main(int argc, char* argv[])
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto open_cl = opencl_manager{};
    open_cl.compile_program(kernel_file);
    open_cl.load_kernel("rotate_image");

    auto&& tga_image = load_tga_image(image_file);

    std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, tga::TGAImage& image, float theta)> fun = rotate_image;
    open_cl.execute_kernel<tga::TGAImage&, const float>("rotate_image", fun, tga_image, std::move(theta));

	/*auto&& tga_image = load_tga_image(image_file);
	rotate_image_seq(tga_image, std::move(theta));*/

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); 
	std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    std::getchar();
    return 0;
}