import numpy as np
import pyopencl as cl

# OpenCL kernel code
kernel_code = open("../tile.cl", "r").read()

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
