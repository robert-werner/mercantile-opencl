# PyOpenCL kernel code
kernel_code = """
__kernel void generate_tiles(
    __global const double *bbox, 
    __global const int *zoom_levels, 
    __global int *tiles,
    const int num_zoom_levels,
    const int max_tile_count_per_zoom
) {
    int gid = get_global_id(0);
    int zoom_index = gid / max_tile_count_per_zoom;
    if (zoom_index >= num_zoom_levels) return;

    int zoom = zoom_levels[zoom_index];
    double lon_min = bbox[0];
    double lat_min = bbox[1];
    double lon_max = bbox[2];
    double lat_max = bbox[3];

    // Convert longitude and latitude to tile coordinates
    int x_min = (int)((lon_min + 180.0) / 360.0 * (1 << zoom));
    int y_min = (int)((1.0 - log(tan(lat_min * M_PI / 180.0) + 1.0 / cos(lat_min * M_PI / 180.0)) / M_PI) / 2.0 * (1 << zoom));
    int x_max = (int)((lon_max + 180.0) / 360.0 * (1 << zoom));
    int y_max = (int)((1.0 - log(tan(lat_max * M_PI / 180.0) + 1.0 / cos(lat_max * M_PI / 180.0)) / M_PI) / 2.0 * (1 << zoom));

    // Calculate the total number of tiles for this zoom level
    int total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1);

    // Find the tile index within the current zoom level
    int tile_index = gid % max_tile_count_per_zoom;

    // Skip if the index is outside the range for the current zoom level
    if (tile_index >= total_tiles) return;

    // Calculate the x and y coordinates for the current tile index
    int x = x_min + (tile_index % (x_max - x_min + 1));
    int y = y_min + (tile_index / (x_max - x_min + 1));

    // Write the results to the output array
    tiles[gid * 3] = zoom;
    tiles[gid * 3 + 1] = x;
    tiles[gid * 3 + 2] = y;
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
bbox = np.array([-180, -85.05112878, 180, 85.05112878], dtype=np.float64)  # Whole world
zoom_levels = np.array(range(1, 5), dtype=np.int32)  # Zoom levels 1 to 4

# Calculate the maximum number of tiles for each zoom level
max_tile_count_per_zoom = 0
for zoom in zoom_levels:
    x_min = int((bbox[0] + 180.0) / 360.0 * (1 << zoom))
    y_min = int((1.0 - np.log(np.tan(np.radians(bbox[1])) + 1.0 / np.cos(np.radians(bbox[1]))) / np.pi) / 2.0 * (1 << zoom))
    x_max = int((bbox[2] + 180.0) / 360.0 * (1 << zoom))
    y_max = int((1.0 - np.log(np.tan(np.radians(bbox[3])) + 1.0 / np.cos(np.radians(bbox[3]))) / np.pi) / 2.0 * (1 << zoom))
    tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    if tile_count > max_tile_count_per_zoom:
        max_tile_count_per_zoom = tile_count

total_tile_count = max_tile_count_per_zoom * len(zoom_levels)

# Create buffers
bbox_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bbox)
zoom_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=zoom_levels)
tiles_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, total_tile_count * 3 * 4)  # Each tile has 3 integers (zoom, x, y)

# Compile the kernel
program = cl.Program(context, kernel_code).build()

# Execute the kernel
num_zoom_levels = np.int32(len(zoom_levels))
program.generate_tiles(queue, (total_tile_count,), None, bbox_buf, zoom_buf, tiles_buf, num_zoom_levels, np.int32(max_tile_count_per_zoom))

# Read back the results
tiles = np.empty(total_tile_count * 3, dtype=np.int32)
cl.enqueue_copy(queue, tiles, tiles_buf)

# Reshape the result to make it more readable
tiles = tiles.reshape(-1, 3)

# Filter out the zeros (unused slots)
tiles = tiles[np.any(tiles != 0, axis=1)]

# Print the results
for tile in tiles:
    print(f"Zoom level: {tile[0]}, X: {tile[1]}, Y: {tile[2]}")
