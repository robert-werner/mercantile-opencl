import pyopencl as cl

# Get all available platforms (e.g., CPU, GPU)
platforms = cl.get_platforms()

# Iterate over the platforms and print their attributes and devices
for i, platform in enumerate(platforms):
    print(f"Platform {i}: {platform.name}")
    print(f"Platform Vendor: {platform.vendor}")
    print(f"Platform Version: {platform.version}")

    # Get all devices available for the platform
    devices = platform.get_devices()

    # Iterate over the devices and print their attributes
    for j, device in enumerate(devices):
        print(f"\tDevice {j}: {device.name}")
        print(f"\tDevice Type: {cl.device_type.to_string(device.type)}")
        print(f"\tDevice Vendor: {device.vendor}")
        print(f"\tDevice Version: {device.version}")
        print(f"\tDriver Version: {device.driver_version}")
        print()
