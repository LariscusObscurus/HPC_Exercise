#pragma once
//#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <functional>

class opencl_manager
{
    cl::Context context_;
    std::vector<cl::Device> devices_;
    std::unordered_map<std::string, cl::Kernel> kernels_;
    cl::CommandQueue queue_;

    cl_int err_ = CL_SUCCESS;
    cl::Program program_;

public:
    opencl_manager();

    void compile_program(const std::string& kernel_file);
    void load_kernel(const std::string& kernel_name);

    template<typename... Params>
    void execute_kernel(const std::string& kernel_name, std::function<void(cl::Context&, cl::CommandQueue&, cl::Kernel& , Params...)>& function, Params&&... args)
    {
        auto kernel = kernels_.find(kernel_name);
        if (kernel == kernels_.end())
            throw std::runtime_error("kernel:" + kernel_name + " does not exist.");
        function(context_, queue_, kernel->second, std::forward<Params>(args)...);
    }

    cl::Kernel get_kernel(const std::string& kernel_name) const
    {
        return kernels_.find(kernel_name)->second;
    }

    int get_error() const { return err_; }

    int get_max_workgroup_size()
    {
        //TODO: Make adjustable
        return devices_[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    }
};