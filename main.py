# PyOpenCL kernel code
from numba import prange

kernel_code = """
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
"""

import pyopencl as cl
import numpy as np

# Set up OpenCL context and queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Define the bounding box and zoom levels
bbox = np.array([30.2046614254646641, 51.9305794595394374, 49.9243936511685149, 61.4949180004405278],
                dtype=np.float64)  # Whole world
zoom_levels = np.array(range(21), dtype=np.int32)  # Zoom levels 1 to 4

# Create buffers
bbox_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bbox)
zoom_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=zoom_levels)
tiles_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, zoom_levels.nbytes * 5)

# Compile the kernel
program = cl.Program(context, kernel_code).build()

# Execute the kernel
num_zoom_levels = np.int32(len(zoom_levels))
program.generate_tiles(queue, zoom_levels.shape, None, bbox_buf, zoom_buf, tiles_buf, num_zoom_levels)

# Read back the results
tiles = np.empty(zoom_levels.size * 5, dtype=np.int32)
cl.enqueue_copy(queue, tiles, tiles_buf)

# Reshape the result to make it more readable
tiles = tiles.reshape(-1, 5)

print(tiles)

# Print the resultsl
for tile in tiles:
    print(f"Zoom level: {tile[0]}, X range: {tile[1]}-{tile[3]}, Y range: {tile[4]}-{tile[2]}")
