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
    __local int * temp,
    const int items)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int depth = 1;

    temp[2 * local_id] = input[2 * global_id];
    temp[2 * local_id + 1] = input[2 * global_id + 1];

    //upsweep
    for (int stride = items >> 1; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < stride) {
            int i = depth * (2 * local_id + 1) - 1;
            int j = depth * (2 * local_id + 2) - 1;
            temp[j] += temp[i];
        }

        depth <<= 1;
    }

    if (local_id == 0) {
        temp[items - 1] = 0;
    }

    //downsweep
    for (int stride = 1; stride < items; stride <<= 1) {
        depth >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < stride) {
            int i = depth * (2 * local_id + 1) - 1;
            int j = depth * (2 * local_id + 2) - 1;

            int t = temp[i];
            temp[i] = temp[j];
            temp[j] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output[2 * global_id] = temp[2 * local_id];
    output[2 * global_id + 1] = temp[2 * local_id + 1];
}