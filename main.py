import pyopencl as cl
import numpy as np
import math

# OpenCL kernel code
kernel_code = """
#define EPSILON 1e-14

typedef struct {
    int x;
    int y;
    int z;
} Tile;

Tile tile(float lng, float lat, int zoom) {
    float x = (lng + 180.0f) / 360.0f;
    float y = (1.0f - log(tan(radians(lat)) + (1.0f / cos(radians(lat)))) / M_PI_F) / 2.0f;

    int Z2 = (int)pow(2.0f, (float)zoom);

    int xtile, ytile;
    if (x <= 0.0f)
        xtile = 0;
    else if (x >= 1.0f)
        xtile = Z2 - 1;
    else
        xtile = floor((x + EPSILON) * Z2);

    if (y <= 0.0f)
        ytile = 0;
    else if (y >= 1.0f)
        ytile = Z2 - 1;
    else
        ytile = floor((y + EPSILON) * Z2);

    return (Tile){xtile, ytile, zoom};
}

__kernel void tile_kernel(__global float* lng, __global float* lat, __global int* zoom, __global Tile* result) {
    int gid = get_global_id(0);
    result[gid] = tile(lng[gid], lat[gid], zoom[gid]);
}
"""

# Create a PyOpenCL context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Compile the OpenCL kernel
program = cl.Program(context, kernel_code).build()

# Create input and output arrays
lng_np = np.array([55.754150], dtype=np.float32)
lat_np = np.array([37.602215], dtype=np.float32)
zoom_np = np.array([10], dtype=np.int32)
result_np = np.empty(1, dtype=[('x', np.int32), ('y', np.int32), ('z', np.int32)])

# Create OpenCL buffers
lng_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=lng_np)
lat_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=lat_np)
zoom_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=zoom_np)
result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result_np.nbytes)

# Run the OpenCL kernel
program.tile_kernel(queue, lng_np.shape, None, lng_buf, lat_buf, zoom_buf, result_buf)

# Copy the result back to the host
cl.enqueue_copy(queue, result_np, result_buf)

# Print the result
print(result_np)
