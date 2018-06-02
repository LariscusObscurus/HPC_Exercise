#pragma once
#include <CL/cl.hpp>
#include <functional>

class opencl_manager
{
    cl::Context context_;
    std::vector<cl::Device> devices_;
    cl::Kernel kernel_;
    cl::CommandQueue queue_;

    cl_int err_ = CL_SUCCESS;

public:
    opencl_manager();

    void load_kernel(const std::string& kernel_file, const std::string& kernel_name);

    template<typename... Params>
    void execute_kernel(std::function<void(cl::Context&, cl::CommandQueue&, cl::Kernel, Params...)>& function, Params&&... args)
    {
        function(context_, queue_, kernel_, args...);
    }

    int get_error() const { return err_; }
};