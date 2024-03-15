__kernel void mercantile_tile(
    __global const double *lng,
    __global const double *lat,
    const unsigned int zoom,
    __global int *tile_x,
    __global int *tile_y,
    const unsigned int count) {

    const double pi = 3.1415926535897932384626433832795;
    int i = get_global_id(0);
    if (i < count) {
        double lat_rad = radians(lat[i]);
        double n = pow(2.0, zoom);
        int x = (int)((lng[i] + 180.0) / 360.0 * n);
        int y = (int)((1.0 - log(tan(lat_rad) + 1.0 / cos(lat_rad)) / pi) / 2.0 * n);

        tile_x[i] = x;
        tile_y[i] = y;
    }
}

inline double radians(double degrees) {
    return degrees * (pi / 180.0);
}
