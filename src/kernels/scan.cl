__kernel
void single_workgroup_prefixsum(
    __global const int* restrict input,
    __global int* restrict output,
    const int n)
{
    output[0] = 0;
    for (int i = 0; i < n; i++) {
        output[i + 1] = output[i] + input[i];
    }
}

__kernel void naive_parallel_prefixsum(__global int* input,
    __global int* output,
    __local int* temp_a,
    __local int* temp_b)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int size = get_local_size(0);

    temp_a[local_id] = temp_b[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 1; stride < size; stride <<= 1) {

        if (local_id >= stride) {
            temp_b[local_id] = temp_a[local_id] + temp_a[local_id - stride];
        }
        else {
            temp_b[local_id] = temp_a[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local int* tmp = temp_a;
        temp_a = temp_b;
        temp_b = tmp;
    }
    output[global_id] = temp_a[local_id];
}


__kernel void blelloch_scan(__global const int* input,
    __global int* output,
    __global int* group_sum,
    __local int * temp,
    const int items)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint group_id = get_group_id(0);
    uint group_size = get_local_size(0);

    uint depth = 1;

    if (global_id >= items) return;

    temp[local_id] = input[global_id];

    //upsweep
    for (uint stride = group_size >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < stride) {
            uint i = depth * (2 * local_id + 1) - 1;
            uint j = depth * (2 * local_id + 2) - 1;
            temp[j] += temp[i];
        }

        depth <<= 1;
    }

    if (local_id == 0) {
        group_sum[group_id] = temp[group_size - 1];
        temp[group_size - 1] = 0;
    }

    //downsweep
    for (uint stride = 1; stride < group_size; stride <<= 1) {

        depth >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < stride) {
            uint i = depth * (2 * local_id + 1) - 1;
            uint j = depth * (2 * local_id + 2) - 1;

            int t = temp[j];
            temp[j] += temp[i];
            temp[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output[global_id] = temp[local_id];
}

__kernel void add_groups(
    __global int* input,
    __global int* output,
    __global int* sums)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    output[global_id] = input[global_id] + sums[group_id];
}