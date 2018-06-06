#include "CL/cl.hpp"
#include <opencl_manager.h>
#include <string>
#include <functional>
#include <iostream>
#include <cassert>
#include <chrono>
#include <prefix_sum.h>
#include <constants/prefix_sum_constants.h>

template<typename T>
void print_vector(std::vector<T> vector)
{
    std::cout << "[(size: " << vector.size() << ") ";
    for (auto& it : vector)
    {
        std::cout << it << " ";
    }
    std::cout << "]" << std::endl;
}

bool check_result(const std::vector<int> expected_data, const std::vector<int> actual_data)
{
    for (auto i = 0; i < expected_data.size(); ++i)
    {
        if (expected_data[i] != actual_data[i])
        {
            std::cout << "At pos " << i << " Result was: " << actual_data[i] << " Should be: " << expected_data[i] << std::endl;

            return false;
        }
    }
    std::cout << "Result OK. " << std::endl;

    return true;
}

template<typename T>
int measure_runtime(std::function<T> func) {
    const auto start_time = std::chrono::steady_clock::now();

    func();

    const auto end_time = std::chrono::steady_clock::now();
    const auto diff = end_time - start_time;

    return std::chrono::duration<double, std::milli>(diff).count();
}

void test_sequential(const std::vector<int>& test_data)
{
    std::cout << "Testing sequential sum: " << std::endl;

    std::function<std::vector<int>()> test_function = [&] { return sequential_scan_inclusive(test_data); };;
	auto result = measure_runtime(test_function);

	std::cout << "Elapsed time: " << result << " ms" << std::endl;
}

void test_workefficient_gpu(opencl_manager& open_cl, const std::vector<int>& test_data, const std::vector<int>& expected_data)
{
    std::cout << "Testing Blelloch sum: " << std::endl;
    auto output = std::vector<int>(test_data.size());

    std::function<void()> test_function = [&]() {
        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>&, std::vector<int>&, const opencl_manager&)> workefficient_scan = gpu_workefficient_prefixsum;

        open_cl.execute_kernel<const std::vector<int>&, std::vector<int>&, const opencl_manager&>("blelloch_scan", workefficient_scan, test_data, output, open_cl);
    };

	auto result = measure_runtime(test_function);

	std::cout << (std::equal(output.begin(), output.end(), expected_data.begin()) ? "Result is correct" : "Result is INCORRECT") << std::endl;
	std::cout << "Elapsed time: " << result << " ms" << std::endl;

}

void test_naive_gpu(opencl_manager& open_cl, const std::vector<int>& test_data, const std::vector<int>& expected_data)
{
    std::cout << "Testing Naive sum: " << std::endl;
    auto output = std::vector<int>(test_data.size());

    std::function<void()> test_function = [&]() {
        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>&, std::vector<int>&)> fun = gpu_prefixsum;

        open_cl.execute_kernel("naive_parallel_prefixsum", fun, test_data, output); 
    };

	auto result = measure_runtime(test_function);

	std::cout << (std::equal(output.begin(), output.end(), expected_data.begin()) ? "Result is correct" : "Result is INCORRECT") << std::endl;
	std::cout << "Elapsed time: " << result << " ms" << std::endl;
}

void test_naive_gpu2(opencl_manager& open_cl, const std::vector<int>& test_data, const std::vector<int>& expected_data)
{
    std::cout << "Testing Naive sum: " << std::endl;
    auto output = std::vector<int>(test_data.size());

    std::function<void()> test_function = [&]() {
        std::function<void(cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel, const std::vector<int>&, std::vector<int>&)> fun = gpu_prefixsum2;

        open_cl.execute_kernel("naive_parallel_prefixsum2", fun, test_data, output); 
    };

    auto result = measure_runtime(test_function);

	std::cout << (std::equal(output.begin(), output.end(), expected_data.begin()) ? "Result is correct" : "Result is INCORRECT") << std::endl;
    std::cout << "Elapsed time: " << result << " ms" << std::endl;
}

int main(int argc, char* argv[])
{
    try
    {
        auto open_cl = opencl_manager{};
        open_cl.compile_program("scan.cl");
        open_cl.load_kernel("single_workgroup_prefixsum");
        open_cl.load_kernel("blelloch_scan");
        open_cl.load_kernel("add_groups");
        open_cl.load_kernel("naive_parallel_prefixsum");
        open_cl.load_kernel("naive_parallel_prefixsum2");

        const auto items = 1024 * 1024 * 1024;

        //Fill test vector
        auto test = std::vector<int>{};
        random_fill_vector(items, test);


        const auto expected_data_inclusive = sequential_scan_inclusive(test);
        const auto expected_data_exclusive = sequential_scan_exclusive(test);

        test_sequential(test);

        if (items <= 1024 * 4) {
            test_naive_gpu(open_cl, test, expected_data_inclusive);
            test_naive_gpu2(open_cl, test, expected_data_inclusive);
        } else
        {
            std::cout << "Too big" << std::endl;
        }
		test_workefficient_gpu(open_cl, test, expected_data_exclusive);
    }
    catch (std::runtime_error ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::getchar();
    return 0;
}

