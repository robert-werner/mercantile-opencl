import numpy as np
import pyopencl as cl

# OpenCL kernel code
kernel_code = """
#define LL_EPSILON 1e-12

typedef struct {
    long x;
    long y;
    long z;
} Tile;

__kernel void generate_tiles_kernel(__global Tile *result, double w, double s, double e, double n, long zoom, long num_tiles, long ul_tile_x, long ul_tile_y, long lr_tile_x, long lr_tile_y) {
    int gid = get_global_id(0);
    if (gid < num_tiles) {
        result[gid].x = gid % (lr_tile_x - ul_tile_x + 1) + ul_tile_x;
        result[gid].y = gid / (lr_tile_x - ul_tile_x + 1) + ul_tile_y;
        result[gid].z = zoom;
    }
}
"""


def truncate_lnglat(lng, lat):
    # Implement the truncate_lnglat function
    # Example implementation:
    lng = max(min(lng, 180.0), -180.0)
    lat = max(min(lat, 90.0), -90.0)
    return lng, lat


def tile(lng, lat, zoom):
    # Implement the tile function
    # Example implementation:
    n = 2.0 ** zoom
    xtile = int((lng + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(np.radians(lat)) + (1 / np.cos(np.radians(lat)))) / np.pi) / 2.0 * n)
    return xtile, ytile


def generate_tiles_opencl(w, s, e, n, zoom, truncate=False):
    if truncate:
        w, s = truncate_lnglat(w, s)
        e, n = truncate_lnglat(e, n)

    ul_tile = tile(w, n, zoom)
    lr_tile = tile(e - 1e-12, s + 1e-12, zoom)

    num_tiles = (lr_tile[0] - ul_tile[0] + 1) * (lr_tile[1] - ul_tile[1] + 1)

    # Create a PyOpenCL context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Define the Tile structure
    tile_struct = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])

    # Allocate memory for the output buffer
    result_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_tiles * tile_struct.itemsize)

    # Build the OpenCL program
    prg = cl.Program(ctx, kernel_code).build()

    # Run the OpenCL kernel
    prg.generate_tiles_kernel(queue, (num_tiles,), None, result_buf, np.float64(w), np.float64(s), np.float64(e),
                              np.float64(n), np.int64(zoom), np.int64(num_tiles), np.int64(ul_tile[0]),
                              np.int64(ul_tile[1]), np.int64(lr_tile[0]), np.int64(lr_tile[1]))

    # Copy the output from the device to the host
    result = np.empty(num_tiles, dtype=tile_struct)
    cl.enqueue_copy(queue, result, result_buf)

    return result


# Example usage
w, s, e, n = 36.7135257279520175, 55.2526640997541278, 38.6930278499606359, 56.2189907271275899
zoom = 10
truncate = True

tiles = generate_tiles_opencl(w, s, e, n, zoom, truncate)

print("Generated Tiles:")
for tile in tiles:
    print(f"X: {tile['x']}, Y: {tile['y']}, Zoom: {tile['z']}")
