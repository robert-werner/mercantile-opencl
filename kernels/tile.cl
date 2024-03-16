#define EPSILON 1e-14

typedef struct {
    int x;
    int y;
    int z;
} Tile;

float _xToLng(float x) {
    return x * 360.0 - 180.0;
}

float _yToLat(float y) {
    float y2 = 180.0 - y * 360.0;
    return 360.0 / M_PI * atan(exp(y2 * M_PI / 180.0)) - 90.0;
}

Tile tile(float lng, float lat, int zoom) {
    float x = (lng + 180.0) / 360.0;
    float y = (1.0 - log(tan(lat * M_PI / 180.0) + 1.0 / cos(lat * M_PI / 180.0)) / M_PI) / 2.0;

    int Z2 = (int)pow((float)2, (float)zoom);

    int xtile, ytile;

    if (x <= 0.0f) {
        xtile = 0;
    } else if (x >= 1.0f) {
        xtile = Z2 - 1;
    } else {
        xtile = floor((x + EPSILON) * Z2);
    }

    if (y <= 0.0f) {
        ytile = 0;
    } else if (y >= 1.0f) {
        ytile = Z2 - 1;
    } else {
        ytile = floor((y + EPSILON) * Z2);
    }

    Tile t = {xtile, ytile, zoom};
    return t;
}

__kernel void tile_kernel(__global float* lngs, __global float* lats, __global int* zooms, __global Tile* output) {
    int gid = get_global_id(0);
    output[gid] = tile(lngs[gid], lats[gid], zooms[gid]);
}