#include "opencl_manager.h"
#include <iostream>
#include <fstream>

opencl_manager::opencl_manager()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty())
    {
        throw std::runtime_error("No OpenCL platform found");
    }
    else
    {
        std::cout << "Found " << platforms.size() << " platform(s)" << std::endl;
    }

    for (auto& platform : platforms)
    {
        std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    //Platforms might need ajustment depending on the PC

    context_ = cl::Context(CL_DEVICE_TYPE_GPU, properties);

    devices_ = context_.getInfo<CL_CONTEXT_DEVICES>();
}

void opencl_manager::load_kernel(const std::string& kernel_file, const std::string& kernel_name)
{
    std::ifstream source_file(kernel_file);

    if (!source_file)
    {
        throw std::runtime_error("kernel source file " + kernel_file + " not found!");
    }

    std::string source_code(
        std::istreambuf_iterator<char>(source_file),
        (std::istreambuf_iterator<char>()));

    const auto source = cl::Program::Sources(1, std::make_pair(source_code.c_str(), source_code.length() + 1));
    auto program = cl::Program(context_, source);

    const auto rv = program.build(devices_);
    if (rv != CL_SUCCESS)
    {
        const auto build_info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices_[0]);
        std::cerr << build_info << std::endl << std::endl;
        std::getchar();
        throw std::runtime_error("Compiling program failed.");
    }

    queue_ = cl::CommandQueue(context_, devices_[0], 0, &err_);
    kernel_ = cl::Kernel(program, kernel_name.c_str(), &err_);
    if (err_ != CL_SUCCESS)
    {
        throw std::runtime_error("Kernel creation failed.");
    }
}
