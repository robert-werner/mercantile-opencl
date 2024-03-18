import numpy as np
import pyopencl as cl

# OpenCL kernel code
kernel_code = """
typedef struct {
    int x;
    int y;
    int z;
} Tile;

__kernel void quadkey_to_tile(__global const char *quadkeys, __global Tile *tiles, int num_quadkeys) {
    int gid = get_global_id(0);
    if (gid < num_quadkeys) {
        int offset = gid * 256;
        int len = 0;
        while (quadkeys[offset + len] != '\\0') {
            len++;
        }
        if (len == 0) {
            tiles[gid].x = 0;
            tiles[gid].y = 0;
            tiles[gid].z = 0;
        } else {
            int xtile = 0, ytile = 0;
            for (int i = len - 1; i >= 0; i--) {
                int mask = 1 << (len - 1 - i);
                char digit = quadkeys[offset + i];
                if (digit == '1') {
                    xtile |= mask;
                } else if (digit == '2') {
                    ytile |= mask;
                } else if (digit == '3') {
                    xtile |= mask;
                    ytile |= mask;
                }
            }
            tiles[gid].x = xtile;
            tiles[gid].y = ytile;
            tiles[gid].z = len;
        }
    }
}
"""

# Create a PyOpenCL context and command queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Define the Tile structure
tile_struct = np.dtype([("x", np.int32), ("y", np.int32), ("z", np.int32)])

# Create sample input data
quadkeys = np.array(["1230013310"])
num_quadkeys = quadkeys.shape[0]

# Convert quadkeys to a contiguous buffer of chars
quadkeys_buffer = np.zeros(num_quadkeys * 256, dtype=np.int8)
for i, quadkey in enumerate(quadkeys):
    quadkeys_buffer[i * 256 : i * 256 + len(quadkey)] = [ord(c) for c in quadkey]

# Allocate memory for input and output buffers
quadkeys_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=quadkeys_buffer)
tiles_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_quadkeys * tile_struct.itemsize)

# Build the OpenCL program
prg = cl.Program(ctx, kernel_code).build()

# Run the OpenCL kernel
prg.quadkey_to_tile(queue, (num_quadkeys,), None, quadkeys_buf, tiles_buf, np.int32(num_quadkeys))

# Copy the output from the device to the host
tiles = np.empty(num_quadkeys, dtype=tile_struct)
cl.enqueue_copy(queue, tiles, tiles_buf)

# Print the tiles
print("Tiles:")
for tile in tiles:
    print(f"x: {tile['x']}, y: {tile['y']}, z: {tile['z']}")
