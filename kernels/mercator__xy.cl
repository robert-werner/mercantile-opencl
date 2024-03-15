__kernel void mercantile_xy(__global const double *lng, __global const double *lat, __global double *x, __global double *y, const unsigned int count) {
    const double origin_shift = 2.0 * M_PI * 6378137.0 / 2.0;
    int i = get_global_id(0);
    if (i < count) {
        double lon = lng[i];
        double lat_rad = radians(lat[i]);
        x[i] = lon * origin_shift / 180.0;
        y[i] = log(tan((M_PI / 4.0) + (lat_rad / 2.0))) * origin_shift / M_PI;
    }
}