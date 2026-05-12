import numpy as np
import matplotlib.pyplot as plt


import raysim.cuda as rs

# Step 1: Create materials registry
materials = rs.Materials()

# Step 2: Create a world with water background
world = rs.World("water")

# Step 3: Add a simple sphere (currently required for simulator to work)
material_id = materials.get_index("water")  # High contrast material
sphere = rs.Sphere(
    np.array([0, 0, 100], dtype=np.float32),  # Center: 100mm in front of probe
    20,                                       # Radius: 20mm
    material_id                               # Material ID
)
world.add(sphere)

# Step 4: Create a simple probe
pose = rs.Pose(
    position=[0., 0., 0.],   # Probe position in world
    rotation=[0., 0., 0.]    # Probe orientation
)
probe = rs.LinearArrayProbe(pose,
                       frequency=2, # MHz
                       )

# Step 5: Create the simulator
simulator = rs.RaytracingUltrasoundSimulator(world, materials)

# Step 6: Configure and run the simulation
sim_params = rs.SimParams()
sim_params.conv_psf = True
sim_params.b_mode_size = (1500, 1500)
image = simulator.simulate(probe, sim_params)

# Step 7: Display the result with proper scaling
# Normalize the image for display (typical ultrasound dynamic range)
min_val = -60.0  # dB
max_val = 0.0    # dB
normalized_image = np.clip((image - min_val) / (max_val - min_val), 0, 1)

min_x = simulator.get_min_x()
max_x = simulator.get_max_x()
min_z = simulator.get_min_z()
max_z = simulator.get_max_z()


plt.figure(figsize=(10, 8))
plt.imshow(
    normalized_image,
    cmap='gray',
    extent=[min_x, max_x, min_z, max_z],  # Use physical coordinates
    aspect='auto'
)
plt.title("Your First Ultrasound Simulation")
plt.xlabel("Width (mm)")
plt.ylabel("Depth (mm)")
plt.colorbar(label="Normalized Intensity")

# Add annotation showing the sphere location
plt.savefig("Getting_Started_Chapter_1.png")
plt.show()

print("Success! You've created your first ultrasound simulation!")
print(f"Image dimensions: {image.shape}")
print(f"Value range: {image.min():.2f} to {image.max():.2f}")
print(f"Sphere material: water (same as background - geometric interface)")