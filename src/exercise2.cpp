#include "CL/cl.hpp"
#include <string>
#include <functional>
#include <opencl_manager.h>
#include <random>
#include <iostream>

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

template <size_t size>
void fill_vector(std::vector<int>& v)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, size);
    generate_n(back_inserter(v), size, bind(dist, gen));
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
    kernel.setArg(2, (int)input.size());

    const auto offset = cl::NDRange(0);
    const auto global = cl::NDRange(input.size());

    const auto rv = queue.enqueueNDRangeKernel(kernel, offset, global);
    if (rv != CL_SUCCESS)
    {
        throw std::runtime_error("Could not enqueue kernel. Return value was:  " + std::to_string(rv));
    }

    auto event = cl::Event{};
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_buffer_size, &output[0], nullptr, &event);

    queue.finish();
    event.wait();
}

int main(int argc, char* argv[])
{
    try
    {

        auto test = std::vector<int>{ 1, 2, 3, 4, 5 };
        auto result = sequential_scan(test);
        std::cout << "Test Vector: " << std::endl;
        print_vector(test);

        std::cout << "Result: " << std::endl;
        print_vector(result);

        auto open_cl = opencl_manager{};
        open_cl.compile_program("scan.cl");
        open_cl.load_kernel("single_workgroup_prefixsum");

        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, std::vector<int>&, std::vector<int>&)> fun = gpu_prefixsum;

        auto output = std::vector<int>(test.size() + 1); //Add one for the leading zero.
        open_cl.execute_kernel("single_workgroup_prefixsum", fun, test, output);

        std::cout << "GPU Result: " << std::endl;
        print_vector(output);
    }
    catch (std::runtime_error ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::getchar();
    return 0;
}
