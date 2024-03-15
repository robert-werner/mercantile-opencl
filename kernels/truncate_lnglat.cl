__kernel void truncate_lnglat(__global float *lng, __global float *lat, const unsigned int count) {
    int i = get_global_id(0);
    if (i < count) {
        if (lng[i] < -180.0f) lng[i] = -180.0f;
        else if (lng[i] > 180.0f) lng[i] = 180.0f;

        if (lat[i] < -85.051129f) lat[i] = -85.051129f;
        else if (lat[i] > 85.051129f) lat[i] = 85.051129f;
    }
}