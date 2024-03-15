__kernel void generate_tiles(
    __global const double *bbox,
    __global const int *zoom_levels,
    __global int *tiles,
    const int num_zoom_levels
) {
    int gid = get_global_id(0);
    int zoom = zoom_levels[gid];
    double lon_min = bbox[0];
    double lat_min = bbox[1];
    double lon_max = bbox[2];
    double lat_max = bbox[3];

    // Convert longitude and latitude to tile coordinates
    int x_min = (int)((lon_min + 180.0) / 360.0 * (1 << zoom));
    int y_min = (int)((1.0 - log(tan(lat_min * M_PI / 180.0) + 1.0 / cos(lat_min * M_PI / 180.0)) / M_PI) / 2.0 * (1 << zoom));
    int x_max = (int)((lon_max + 180.0) / 360.0 * (1 << zoom));
    int y_max = (int)((1.0 - log(tan(lat_max * M_PI / 180.0) + 1.0 / cos(lat_max * M_PI / 180.0)) / M_PI) / 2.0 * (1 << zoom));

    // Write the results to the output array
    int index = gid * 5;
    tiles[index] = zoom;
    tiles[index + 1] = x_min;
    tiles[index + 2] = y_min;
    tiles[index + 3] = x_max;
    tiles[index + 4] = y_max;
}