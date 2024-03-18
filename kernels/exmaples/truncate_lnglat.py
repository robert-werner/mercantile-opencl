import numpy as np
import pyopencl as cl

# OpenCL kernel code
kernel_code = """
__kernel void truncate_lnglat_kernel(__global float *lng, __global float *lat, __global float *output) {
    int gid = get_global_id(0);

    float lng_value = lng[gid];
    float lat_value = lat[gid];

    if (lng_value > 180.0f)
        lng_value = 180.0f;
    else if (lng_value < -180.0f)
        lng_value = -180.0f;

    if (lat_value > 90.0f)
        lat_value = 90.0f;
    else if (lat_value < -90.0f)
        lat_value = -90.0f;

    output[gid * 2] = lng_value;
    output[gid * 2 + 1] = lat_value;
}
"""

# Create a PyOpenCL context and command queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Create sample input data
num_points = 10
lng_input = np.random.uniform(-200.0, 200.0, size=num_points).astype(np.float32)
lat_input = np.random.uniform(-100.0, 100.0, size=num_points).astype(np.float32)

# Allocate memory for input and output buffers
lng_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=lng_input)
lat_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=lat_input)
output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=num_points * 2 * np.dtype(np.float32).itemsize)

# Build the OpenCL program
prg = cl.Program(ctx, kernel_code).build()
# Run the OpenCL kernel
prg.truncate_lnglat_kernel(queue, (num_points,), None, lng_buf, lat_buf, output_buf)

# Copy the output from the device to the host
output = np.empty(num_points * 2, dtype=np.float32)
cl.enqueue_copy(queue, output, output_buf)

# Reshape the output into (longitude, latitude) pairs
output = output.reshape((num_points, 2))

# Print the truncated longitude and latitude values
print("Truncated Longitude and Latitude:")
for lng, lat in output:
    print(f"Longitude: {lng}, Latitude: {lat}")
