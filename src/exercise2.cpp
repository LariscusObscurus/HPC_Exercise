#include "CL/cl.hpp"
#include <string>
#include <functional>
#include <opencl_manager.h>
#include <random>
#include <iostream>
#include <cassert>

template<typename T>
void print_vector(std::vector<T> vector)
{
    std::cout << "[ ";
    for (auto& it : vector)
    {
        std::cout << it << " ";
    }
    std::cout << "]" << std::endl;
}

void random_fill_vector(const int size, std::vector<int>& v)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, size);
    generate_n(back_inserter(v), size, bind(dist, gen));
}

void sequential_fill_vector(const int size, std::vector<int>& v)
{
    v.reserve(size);
    for (auto i = 0; i < size; ++i)
    {
        v.push_back(i);
    }
}

//scanl (+) 0 [1..5]
// [0,1,3,6,10,15]
std::vector<int> sequential_scan(std::vector<int> input)
{

    auto result = std::vector<int>{ 0 };
    for (auto& it : input)
    {
        result.emplace_back(result.back() + it);
    }
    return result;
}

void gpu_prefixsum(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>& input, std::vector<int>& output)
{
    const auto input_buffer_size = input.size() * sizeof(int);
    const auto output_buffer_size = output.size() * sizeof(int);

    const auto input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, input_buffer_size);
    const auto output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, output_buffer_size);

    queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_buffer_size, input.data());

    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);

    const int local_size = sizeof(int) * input.size();
    kernel.setArg(2, cl::LocalSpaceArg(cl::Local(local_size)));
    kernel.setArg(3, cl::LocalSpaceArg(cl::Local(local_size)));

    const auto offset = cl::NDRange(0);
    const auto local = cl::NDRange(32);
    const auto global = cl::NDRange(input.size());

    const auto rv = queue.enqueueNDRangeKernel(kernel, offset, global, local);
    if (rv != CL_SUCCESS)
    {
        throw std::runtime_error("Could not enqueue kernel. Return value was:  " + std::to_string(rv));
    }

    auto event = cl::Event{};
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_buffer_size, &output[0], nullptr, &event);

    queue.finish();
    event.wait();
}

void gpu_workefficient_prefixsum(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>& input, std::vector<int>& output)
{
    const auto input_buffer_size = input.size() * sizeof(int);
    const auto output_buffer_size = output.size() * sizeof(int);

    const auto input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, input_buffer_size);
    const auto output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, output_buffer_size);

    auto result = queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_buffer_size, input.data());

    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);

    const int local_size = sizeof(int) * input.size();
    kernel.setArg(2, cl::LocalSpaceArg(cl::Local(local_size)));

    kernel.setArg(3, (int)input.size());

    cl::NDRange global(input.size());
    cl::NDRange local(32);
    cl::NDRange offset(0);

    const auto rv = queue.enqueueNDRangeKernel(kernel, offset, global);

    if (rv != CL_SUCCESS)
    {
        throw std::runtime_error("Could not enqueue kernel. Return value was:  " + std::to_string(rv));
    }

    auto event = cl::Event{};
    result = queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_buffer_size, &output[0], nullptr, &event);

    result = queue.finish();
    result = event.wait();
}

int main(int argc, char* argv[])
{
    try
    {

        auto open_cl = opencl_manager{};
        open_cl.compile_program("scan.cl");
        open_cl.load_kernel("single_workgroup_prefixsum");
        open_cl.load_kernel("blelloch_scan");
        open_cl.load_kernel("naive_parallel_prefixsum");

        //Fill test vector

        auto threads = open_cl.get_max_workgroup_size();
        auto items = threads;

        auto test = std::vector<int>{};
        sequential_fill_vector(items, test);
        auto result = sequential_scan(test);

        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>&, std::vector<int>&)> fun = gpu_prefixsum;

        auto output = std::vector<int>(test.size());
        //open_cl.execute_kernel("single_workgroup_prefixsum", fun, test, output);
        //open_cl.execute_kernel("blelloch_scan", fun, test, output);

        open_cl.execute_kernel("naive_parallel_prefixsum", fun, test, output);
        output.insert(output.begin(), 0);//only for naive because it is a non inclusive scan

        for (auto i = 0; i < test.size(); ++i)
        {
            if (result[i] != output[i])
            {
                std::cout << "At pos " << i << " Result was: " << output[i] << " Should be: " << result[i] << std::endl;

                std::getchar();
                return -1;
            }
        }
        std::cout << "GPU Result OK. " << std::endl;
    }
    catch (std::runtime_error ex)
    {
        std::cout << ex.what() << std::endl;
    }
    std::getchar();
    return 0;
}
