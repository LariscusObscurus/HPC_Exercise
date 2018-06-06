#include "tga.h"
#include <iostream>
#include <CL/cl.hpp>
#include "constants/rotation_constants.h"

typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} pixel;

tga::TGAImage load_tga_image(const std::string& file)
{
    auto image = tga::TGAImage{};
    const auto rv = tga::LoadTGA(&image, file.c_str());
    if (rv <= 0)
    {
        throw std::runtime_error("Image does not exist.");
    }
    return image;
}
 
tga::TGAImage create_tga_image(const tga::TGAImage& original, std::vector<pixel> image_data) {
    auto image = tga::TGAImage();
    image.height = original.height;
    image.width = original.width;
    image.bpp = original.bpp;
    image.type = original.type;

    image.imageData.clear();
    for (auto it : image_data)
    {
        image.imageData.push_back(it.r);
        image.imageData.push_back(it.g);
        image.imageData.push_back(it.b);
    }
    return image;
}

void write_image(const std::string& file, tga::TGAImage&& image) {
    tga::saveTGA(image, file.c_str());
    std::cout << "Image written." << std::endl;
}

void rotate_image(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, tga::TGAImage& image, const float theta)
{
    const auto size = image.height * image.width;
    const auto buffer_size = size * sizeof(pixel);

    auto input_image_data = std::vector<pixel>(size);
    auto output_image_data = std::vector<pixel>(size);

    auto j = 0;
    for (auto i = 0; i < size * 3; i += 3)
    {
        input_image_data[j].r = image.imageData[i];
        input_image_data[j].g = image.imageData[i + 1];
        input_image_data[j].b = image.imageData[i + 2];
        j++;
    }

    //clBuffers
    const auto input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, buffer_size);
    const auto output_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, buffer_size);

    queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, buffer_size, input_image_data.data());

    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    kernel.setArg(2, image.width);
    kernel.setArg(3, image.height);
    kernel.setArg(4, sinf(theta));
    kernel.setArg(5, cosf(theta));

    const auto global = cl::NDRange(image.width, image.height);

    const auto rv = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    if (rv != CL_SUCCESS)
    {
        throw std::runtime_error("Could not enqueue kernel.");
    }

    auto event = cl::Event{};
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, buffer_size, &output_image_data[0], nullptr, &event);

    queue.finish();
    event.wait();

    write_image(output_image_file, create_tga_image(image, output_image_data));
}

void rotate_image_seq(tga::TGAImage& image, const float theta) {
	const auto size = image.height * image.width;

	auto input_image_data = std::vector<pixel>(size);
	auto output_image_data = std::vector<pixel>(size);

	auto j = 0;
	for (auto i = 0; i < size * 3; i += 3)
	{
		input_image_data[j].r = image.imageData[i];
		input_image_data[j].g = image.imageData[i + 1];
		input_image_data[j].b = image.imageData[i + 2];
		j++;
	}
	 
	const float sin_theta = sinf(theta);
	const float cos_theta = cosf(theta);

	for (int ix = 0; ix < image.width; ix++) {
		for (int iy = 0; iy < image.height; iy++) {
			float xpos = (((float)ix - image.width / 2)*cos_theta - ((float)iy - image.height / 2)*sin_theta) + (image.width / 2);
			float ypos = (((float)ix - image.width / 2)*sin_theta + ((float)iy - image.height / 2)*cos_theta) + (image.height / 2);

			if ((((int)xpos >= 0) && ((int)xpos < image.width)) && ((ypos >= 0) && ((int)ypos < image.height)))
			{
				output_image_data[iy*image.width + ix] = input_image_data[(int)(floor(ypos)*image.width + floor(xpos))];
			}
		}
	}
	write_image(output_image_file, create_tga_image(image, output_image_data));
}
