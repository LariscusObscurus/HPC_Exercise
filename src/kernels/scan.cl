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