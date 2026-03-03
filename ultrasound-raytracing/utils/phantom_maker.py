# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phantom maker utility to generate 3D phantom meshes for ultrasound simulation"""

import argparse
import math
import os


def generate_checker_mesh(
    output_dir="mesh",
    board_size=8,
    rect_width=30.0,    # Width of each rectangle (x-axis)
    rect_height=20.0,   # Height of each rectangle (y-axis)
    height=10.0         # Z-height for raised rectangles
):
    """
    Generate a 3D checker pattern mesh with rectangular cells.
    Uses triangulated faces and includes normal vectors.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "calibration_phantom.obj")

    # Calculate total size and starting position
    total_width = board_size * rect_width
    total_height = board_size * rect_height
    start_x = -total_width / 2
    start_y = -total_height / 2

    with open(filename, 'w') as f:
        # Write header
        f.write("# 3D Calibration Phantom Checkerboard\n")
        f.write(f"# Size: {board_size}x{board_size} rectangles\n")
        f.write(f"# Rectangle size: {rect_width}x{rect_height}mm\n")
        f.write(f"# Height: {height}mm\n\n")

        # Add object definition for Assimp
        f.write("o CheckerPattern\n\n")

        # Define normals
        f.write("vn 0.0 0.0 1.0\n")   # up (1)
        f.write("vn 0.0 0.0 -1.0\n")  # down (2)
        f.write("vn 1.0 0.0 0.0\n")   # right (3)
        f.write("vn -1.0 0.0 0.0\n")  # left (4)
        f.write("vn 0.0 1.0 0.0\n")   # forward (5)
        f.write("vn 0.0 -1.0 0.0\n")  # back (6)
        f.write("\n")

        vertex_count = 1
        for row in range(board_size):
            for col in range(board_size):
                x = start_x + (col * rect_width)
                y = start_y + (row * rect_height)
                is_raised = (row + col) % 2 == 0

                # Write vertices for current rectangle
                f.write(f"v {x:.6f} {y:.6f} 0.000000\n")
                f.write(f"v {x + rect_width:.6f} {y:.6f} 0.000000\n")
                f.write(f"v {x + rect_width:.6f} {y + rect_height:.6f} 0.000000\n")
                f.write(f"v {x:.6f} {y + rect_height:.6f} 0.000000\n")

                if is_raised:
                    f.write(f"v {x:.6f} {y:.6f} {height:.6f}\n")
                    f.write(f"v {x + rect_width:.6f} {y:.6f} {height:.6f}\n")
                    f.write(f"v {x + rect_width:.6f} {y + rect_height:.6f} {height:.6f}\n")
                    f.write(f"v {x:.6f} {y + rect_height:.6f} {height:.6f}\n")

                f.write("\n")

                # Write triangulated faces with normals
                if is_raised:
                    v_idx = vertex_count
                    # Bottom face (2 triangles)
                    f.write(f"f {v_idx}//2 {v_idx+1}//2 {v_idx+2}//2\n")
                    f.write(f"f {v_idx}//2 {v_idx+2}//2 {v_idx+3}//2\n")
                    # Top face (2 triangles)
                    f.write(f"f {v_idx+4}//1 {v_idx+5}//1 {v_idx+6}//1\n")
                    f.write(f"f {v_idx+4}//1 {v_idx+6}//1 {v_idx+7}//1\n")
                    # Front face (2 triangles)
                    f.write(f"f {v_idx}//6 {v_idx+1}//6 {v_idx+5}//6\n")
                    f.write(f"f {v_idx}//6 {v_idx+5}//6 {v_idx+4}//6\n")
                    # Right face (2 triangles)
                    f.write(f"f {v_idx+1}//3 {v_idx+2}//3 {v_idx+6}//3\n")
                    f.write(f"f {v_idx+1}//3 {v_idx+6}//3 {v_idx+5}//3\n")
                    # Back face (2 triangles)
                    f.write(f"f {v_idx+2}//5 {v_idx+3}//5 {v_idx+7}//5\n")
                    f.write(f"f {v_idx+2}//5 {v_idx+7}//5 {v_idx+6}//5\n")
                    # Left face (2 triangles)
                    f.write(f"f {v_idx+3}//4 {v_idx}//4 {v_idx+4}//4\n")
                    f.write(f"f {v_idx+3}//4 {v_idx+4}//4 {v_idx+7}//4\n")
                    vertex_count += 8
                else:
                    # Base rectangle (2 triangles)
                    f.write(f"f {vertex_count}//1 {vertex_count+1}//1 {vertex_count+2}//1\n")
                    f.write(f"f {vertex_count}//1 {vertex_count+2}//1 {vertex_count+3}//1\n")
                    vertex_count += 4

                f.write("\n")

    return filename

def generate_sphere_mesh(center, radius, num_segments=32):
    """
    Generate vertices and faces for a sphere.

    Args:
        center: (x, y, z) center coordinates
        radius: sphere radius
        num_segments: number of segments for sphere discretization
    """
    vertices = []
    faces = []
    normals = []

    # Generate vertices
    for i in range(num_segments + 1):
        lat = math.pi * (-0.5 + float(i) / num_segments)
        for j in range(num_segments + 1):
            lon = 2 * math.pi * float(j) / num_segments
            x = center[0] + radius * math.cos(lat) * math.cos(lon)
            y = center[1] + radius * math.cos(lat) * math.sin(lon)
            z = center[2] + radius * math.sin(lat)

            # Vertex position
            vertices.append((x, y, z))
            # Normal vector (normalized direction from center to vertex)
            nx = (x - center[0]) / radius
            ny = (y - center[1]) / radius
            nz = (z - center[2]) / radius
            normals.append((nx, ny, nz))

    # Generate faces
    for i in range(num_segments):
        for j in range(num_segments):
            v1 = i * (num_segments + 1) + j
            v2 = v1 + 1
            v3 = (i + 1) * (num_segments + 1) + j
            v4 = v3 + 1

            # Each quad is split into two triangles
            faces.append((v1 + 1, v2 + 1, v3 + 1))
            faces.append((v2 + 1, v4 + 1, v3 + 1))

    return vertices, faces, normals

def generate_ellipsoid_mesh(center, a, b, c, num_segments=32):
    """
    Generate vertices and faces for an ellipsoid.

    Args:
        center: (x, y, z) center coordinates
        a, b, c: semi-axes lengths
        num_segments: number of segments for discretization
    """
    vertices = []
    faces = []
    normals = []

    # Generate vertices
    for i in range(num_segments + 1):
        lat = math.pi * (-0.5 + float(i) / num_segments)
        for j in range(num_segments + 1):
            lon = 2 * math.pi * float(j) / num_segments
            x = center[0] + a * math.cos(lat) * math.cos(lon)
            y = center[1] + b * math.cos(lat) * math.sin(lon)
            z = center[2] + c * math.sin(lat)

            # Vertex position
            vertices.append((x, y, z))
            # Normal vector (this is an approximation for ellipsoids)
            nx = (x - center[0]) / a
            ny = (y - center[1]) / b
            nz = (z - center[2]) / c
            norm = math.sqrt(nx*nx + ny*ny + nz*nz)
            normals.append((nx/norm, ny/norm, nz/norm))

    # Generate faces (same as sphere)
    for i in range(num_segments):
        for j in range(num_segments):
            v1 = i * (num_segments + 1) + j
            v2 = v1 + 1
            v3 = (i + 1) * (num_segments + 1) + j
            v4 = v3 + 1

            faces.append((v1 + 1, v2 + 1, v3 + 1))
            faces.append((v2 + 1, v4 + 1, v3 + 1))

    return vertices, faces, normals

def generate_sphere_in_oval_phantom(
    output_dir="mesh",
    oval_center=(0, 0, 0),
    oval_size=(60, 40, 30),  # semi-axes lengths in mm
    sphere_radius=10.0,      # radius in mm
    sphere_offset=(0, 0, 0)  # offset from center
):
    """
    Generate a 3D phantom with a sphere inside an oval (ellipsoid).
    Creates separate OBJ files for the sphere and oval, plus a combined file.

    Args:
        output_dir: directory to store output files (will be created if it doesn't exist)
        oval_center: (x, y, z) center of oval
        oval_size: (a, b, c) semi-axes lengths of oval
        sphere_radius: radius of inner sphere
        sphere_offset: (x, y, z) offset of sphere from oval center
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate sphere center
    sphere_center = (
        oval_center[0] + sphere_offset[0],
        oval_center[1] + sphere_offset[1],
        oval_center[2] + sphere_offset[2]
    )

    # Generate meshes
    sphere_verts, sphere_faces, sphere_normals = generate_sphere_mesh(
        sphere_center, sphere_radius
    )
    oval_verts, oval_faces, oval_normals = generate_ellipsoid_mesh(
        oval_center, *oval_size
    )

    # Create filenames
    sphere_filename = os.path.join(output_dir, "sphere.obj")
    oval_filename = os.path.join(output_dir, "oval.obj")
    combined_filename = os.path.join(output_dir, "sphere_in_oval.obj")

    # Write sphere to its own file
    with open(sphere_filename, 'w') as f:
        f.write("# Sphere Object\n")
        f.write(f"# Radius: {sphere_radius}mm\n")
        f.write(f"# Center: {sphere_center}\n\n")
        f.write("o Sphere\n\n")

        # Write vertices
        for v in sphere_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")

        # Write vertex normals
        for n in sphere_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")

        # Write faces
        for face in sphere_faces:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")

    # Write oval to its own file
    with open(oval_filename, 'w') as f:
        f.write("# Oval Object\n")
        f.write(f"# Size: {oval_size[0]}x{oval_size[1]}x{oval_size[2]}mm\n")
        f.write(f"# Center: {oval_center}\n\n")
        f.write("o Oval\n\n")

        # Write vertices
        for v in oval_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")

        # Write vertex normals
        for n in oval_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")

        # Write faces
        for face in oval_faces:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")

    # Write combined file
    with open(combined_filename, 'w') as f:
        # Write header
        f.write("# Sphere in Oval Phantom (Combined)\n")
        f.write(f"# Oval size: {oval_size[0]}x{oval_size[1]}x{oval_size[2]}mm\n")
        f.write(f"# Sphere radius: {sphere_radius}mm\n")
        f.write(f"# Sphere offset: {sphere_offset}\n\n")

        # Write vertices
        for v in sphere_verts + oval_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")

        # Write vertex normals
        for n in sphere_normals + oval_normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")

        # Write sphere faces
        f.write("o Sphere\n")
        for face in sphere_faces:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")
        f.write("\n")

        # Write oval faces
        f.write("o Oval\n")
        vertex_offset = len(sphere_verts)
        normal_offset = len(sphere_normals)
        for face in oval_faces:
            f1 = face[0] + vertex_offset
            f2 = face[1] + vertex_offset
            f3 = face[2] + vertex_offset
            n1 = face[0] + normal_offset
            n2 = face[1] + normal_offset
            n3 = face[2] + normal_offset
            f.write(f"f {f1}//{n1} {f2}//{n2} {f3}//{n3}\n")

    return sphere_filename, oval_filename, combined_filename


def generate_cylinder_mesh(
    output_dir="mesh",
    radius=4.0,
    length=4.0,
    num_segments=129,
    inward_normals=True,
):
    """
    Generate an open cylinder (curved wall only, no caps) for IVUS vessel phantoms.
    Axis is along Y; cross-sections in the xz plane are circles. Units are mm.
    Use inward_normals=True so rays from the lumen (center) hit the front face.
    Default 129 segments avoids alignment with 256 IVUS rays (reduces black bands).

    Args:
        output_dir: Directory to write the OBJ file (default: mesh).
        radius: Cylinder radius in mm (default: 4.0).
        length: Extent along Y in mm; cylinder runs from y=-length/2 to y=length/2 (default: 4.0).
        num_segments: Number of segments around the circumference (default: 129).
        inward_normals: If True, normals point toward the axis (for imaging from inside).

    Returns:
        Path to the written OBJ file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "Cylinder.obj")

    half = length / 2.0
    vertices = []
    normals = []

    # Two rings: y = -half and y = +half
    for ring in (0, 1):
        y = -half if ring == 0 else half
        for i in range(num_segments):
            theta = 2 * math.pi * i / num_segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append((x, y, z))
            # Normal: inward (toward axis) for IVUS so rays from center hit front face
            sign = -1.0 if inward_normals else 1.0
            nx = sign * math.cos(theta)
            nz = sign * math.sin(theta)
            normals.append((nx, 0.0, nz))

    with open(filename, "w") as f:
        f.write("# Open cylinder mesh (IVUS vessel phantom)\n")
        f.write(f"# Radius={radius} mm, length={length} mm, segments={num_segments}\n")
        f.write("# Axis along Y; normals inward for imaging from lumen\n\n")
        f.write("o Cylinder\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")
        # Winding so the inner face (toward axis) is front: swap vertex order vs outward-facing
        for i in range(num_segments):
            i_next = (i + 1) % num_segments
            v0 = i + 1
            v1 = i_next + 1
            v2 = i_next + num_segments + 1
            v3 = i + num_segments + 1
            # (v0, v2, v1) and (v0, v3, v2) so face normal points inward
            f.write(f"f {v0}//{v0} {v2}//{v2} {v1}//{v1}\n")
            f.write(f"f {v0}//{v0} {v3}//{v3} {v2}//{v2}\n")

    return filename


def generate_cylinder_thick_mesh(
    output_dir="mesh",
    inner_radius=3.5,
    outer_radius=4.0,
    length=4.0,
    num_segments=129,
):
    """
    Generate a thick-walled open cylinder (inner and outer surfaces, no caps).
    Writes two OBJ files: Cylinder_inner.obj and Cylinder_outer.obj, both
    with correct winding and normals for IVUS (imaging from inside the lumen).
    Load both meshes into the world to get a wall with thickness.
    Default 129 segments avoids alignment with 256 IVUS rays (reduces black bands).

    Args:
        output_dir: Directory to write the OBJ files (default: mesh).
        inner_radius: Radius of the lumen-facing surface in mm (default: 3.5).
        outer_radius: Radius of the outer surface in mm (default: 4.0).
        length: Extent along Y in mm (default: 4.0).
        num_segments: Number of segments around the circumference (default: 129).

    Returns:
        Tuple (path_inner, path_outer).
    """
    os.makedirs(output_dir, exist_ok=True)
    half = length / 2.0

    # Both surfaces use inward normals so rays from the lumen (and from inside the wall)
    # hit the front face and the tracer records both echoes.
    for name, radius, inward in (
        ("Cylinder_inner", inner_radius, True),
        ("Cylinder_outer", outer_radius, True),
    ):
        vertices = []
        normals = []
        for ring in (0, 1):
            y = -half if ring == 0 else half
            for i in range(num_segments):
                theta = 2 * math.pi * i / num_segments
                x = radius * math.cos(theta)
                z = radius * math.sin(theta)
                vertices.append((x, y, z))
                sign = -1.0 if inward else 1.0
                nx = sign * math.cos(theta)
                nz = sign * math.sin(theta)
                normals.append((nx, 0.0, nz))

        filename = os.path.join(output_dir, f"{name}.obj")
        with open(filename, "w") as f:
            f.write(f"# {name} for IVUS thick vessel wall\n")
            f.write(f"# Radius={radius} mm, length={length} mm\n\n")
            f.write(f"o {name}\n\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write("\n")
            for i in range(num_segments):
                i_next = (i + 1) % num_segments
                v0, v1 = i + 1, i_next + 1
                v2 = i_next + num_segments + 1
                v3 = i + num_segments + 1
                # Inward winding so front face is toward axis (hit from lumen or from inside wall)
                f.write(f"f {v0}//{v0} {v2}//{v2} {v1}//{v1}\n")
                f.write(f"f {v0}//{v0} {v3}//{v3} {v2}//{v2}\n")

    return (
        os.path.join(output_dir, "Cylinder_inner.obj"),
        os.path.join(output_dir, "Cylinder_outer.obj"),
    )


def main():
    parser = argparse.ArgumentParser(description='Generate 3D phantom meshes for ultrasound simulation')
    parser.add_argument(
        'type',
        choices=['checker', 'sphere_in_oval', 'cylinder'],
        help='Type of phantom to generate',
    )
    parser.add_argument('--output', '-o', default='mesh', help='Output directory (default: mesh)')

    # Checker phantom arguments
    parser.add_argument('--board-size', type=int, default=8, help='Number of squares per side (default: 8)')
    parser.add_argument('--rect-width', type=float, default=30.0, help='Width of each rectangle in mm (default: 30.0)')
    parser.add_argument('--rect-height', type=float, default=20.0, help='Height of each rectangle in mm (default: 20.0)')
    parser.add_argument('--checker-height', type=float, default=10.0, help='Height of raised squares in mm (default: 10.0)')

    # Sphere in oval arguments
    parser.add_argument('--oval-size', type=float, nargs=3, default=[60.0, 40.0, 30.0],
                      help='Semi-axes lengths of oval in mm (default: 60.0 40.0 30.0)')
    parser.add_argument('--sphere-radius', type=float, default=10.0,
                      help='Radius of sphere in mm (default: 10.0)')
    parser.add_argument('--sphere-offset', type=float, nargs=3, default=[10.0, 5.0, 0.0],
                      help='Offset of sphere from center in mm (default: 10.0 5.0 0.0)')

    # Cylinder (IVUS vessel) arguments
    parser.add_argument('--cylinder-radius', type=float, default=4.0,
                        help='Cylinder radius in mm (default: 4.0)')
    parser.add_argument('--cylinder-length', type=float, default=4.0,
                        help='Cylinder length along Y in mm (default: 4.0)')
    parser.add_argument('--cylinder-segments', type=int, default=129,
                        help='Number of angular segments (default: 129, avoids IVUS ray alignment)')
    parser.add_argument('--cylinder-thick', action='store_true',
                        help='Generate thick wall: Cylinder_inner.obj and Cylinder_outer.obj')
    parser.add_argument('--cylinder-inner-radius', type=float, default=3.5,
                        help='Inner radius for thick cylinder in mm (default: 3.5)')
    parser.add_argument('--cylinder-outer-radius', type=float, default=4.0,
                        help='Outer radius for thick cylinder in mm (default: 4.0)')

    args = parser.parse_args()

    # Generate the requested phantom
    if args.type == "checker":
        output_file = generate_checker_mesh(
            output_dir=args.output,
            board_size=args.board_size,
            rect_width=args.rect_width,
            rect_height=args.rect_height,
            height=args.checker_height
        )
        print(f"Generated checker phantom: {output_file}")
    elif args.type == "cylinder":
        if getattr(args, 'cylinder_thick', False):
            path_inner, path_outer = generate_cylinder_thick_mesh(
                output_dir=args.output,
                inner_radius=args.cylinder_inner_radius,
                outer_radius=args.cylinder_outer_radius,
                length=args.cylinder_length,
                num_segments=args.cylinder_segments,
            )
            print(f"Generated thick cylinder: {path_inner}, {path_outer}")
        else:
            output_file = generate_cylinder_mesh(
                output_dir=args.output,
                radius=args.cylinder_radius,
                length=args.cylinder_length,
                num_segments=args.cylinder_segments,
            )
            print(f"Generated cylinder mesh: {output_file}")
    else:  # sphere_in_oval
        sphere_file, oval_file, combined_file = generate_sphere_in_oval_phantom(
            output_dir=args.output,
            oval_size=tuple(args.oval_size),
            sphere_radius=args.sphere_radius,
            sphere_offset=tuple(args.sphere_offset)
        )
        print(f"Generated files in directory '{args.output}':")
        print(f"  Sphere mesh: {os.path.basename(sphere_file)}")
        print(f"  Oval mesh: {os.path.basename(oval_file)}")
        print(f"  Combined mesh: {os.path.basename(combined_file)}")

if __name__ == "__main__":
    main()
