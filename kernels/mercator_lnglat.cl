__kernel void mercator_lnglat(__global const float *x, __global const float *y, __global float *lng, __global float *lat, const unsigned int count) {
    int i = get_global_id(0);
    if (i < count) {
        float a = 1.0f / 6378137.0f;
        lng[i] = x[i] * a;
        lat[i] = (2.0f * atan(exp(y[i] * a)) - (float)M_PI_2) * (180.0f / (float)M_PI);
        // Convert radians to degrees
        lng[i] = lng[i] * (180.0f / (float)M_PI);
    }
}
