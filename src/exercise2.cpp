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

int main(int argc, char* argv[])
{
    auto test = std::vector<int>{ 1, 2, 3, 4, 5 };
    auto result = sequential_scan(test);
    std::cout << "Test Vector: " << std::endl;
    print_vector(test);

    std::cout << "Result: " << std::endl;
    print_vector(result);

    //auto open_cl = opencl_manager{};
    //open_cl.load_kernel("scan.cl", "rotate_image");


    //open_cl.execute_kernel();

    std::getchar();
    return 0;
}
