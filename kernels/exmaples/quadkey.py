import numpy as np
import pyopencl as cl

# OpenCL kernel code
kernel_code = """
typedef struct {
    int x;
    int y;
    int z;
} Tile;

__kernel void quadkey_kernel(__global Tile* tiles, __global char* output) {
    int gid = get_global_id(0);
    Tile tile = tiles[gid];
    int xtile = tile.x;
    int ytile = tile.y;
    int zoom = tile.z;

    int offset = gid * (zoom + 1);

    for (int z = zoom; z > 0; z--) {
        int digit = 0;
        int mask = 1 << (z - 1);
        if (xtile & mask) {
            digit += 1;
        }
        if (ytile & mask) {
            digit += 2;
        }
        output[offset + zoom - z] = digit + '0';
    }
    output[offset + zoom] = '\\0';
}
"""

# Create a PyOpenCL context and command queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Define the Tile structure
tile_struct = np.dtype([("x", np.int32), ("y", np.int32), ("z", np.int32)])

# Create sample input data
tiles = np.array([(670, 396, 10)], dtype=tile_struct)
num_tiles = tiles.shape[0]
max_zoom = np.max(tiles["z"])

# Allocate memory for input and output buffers
tiles_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=tiles)
output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_tiles * (max_zoom + 1))

# Build the OpenCL program
prg = cl.Program(ctx, kernel_code).build()

# Run the OpenCL kernel
prg.quadkey_kernel(queue, (num_tiles,), None, tiles_buf, output_buf)

# Copy the output from the device to the host
output = np.empty(num_tiles * (max_zoom + 1), dtype=np.int8)
cl.enqueue_copy(queue, output, output_buf)

# Convert output to strings
quadkeys = []
for i in range(num_tiles):
    offset = i * (max_zoom + 1)
    quadkey = output[offset:offset + max_zoom].tobytes().decode('ascii')
    quadkeys.append(quadkey)

# Print the quadkeys
print("Quadkeys:")
for quadkey in quadkeys:
    print(quadkey)
