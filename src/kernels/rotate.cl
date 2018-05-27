typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} pixel;

__kernel
void rotate_image(
    __global const pixel* src_data,
    __global pixel* dest_data,
    const int width,
    const int height,
    const float sin_theta,
    const float cos_theta)
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    float xpos = (((float)ix - width / 2)*cos_theta - ((float)iy - height / 2)*sin_theta) + (width / 2);
    float ypos = (((float)ix - width / 2)*sin_theta + ((float)iy - height / 2)*cos_theta) + (height / 2);

    if ((((int)xpos >= 0) && ((int)xpos < width)) && (((int)ypos >= 0) && ((int)ypos < height)))
    {
        dest_data[iy*width + ix] = src_data[(int)(floor(ypos)*width + floor(xpos))];
    }
}
