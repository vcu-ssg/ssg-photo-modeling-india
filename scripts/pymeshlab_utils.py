import pymeshlab as ml
from pathlib import Path
import numpy as np
import sys
import open3d as o3d
import numpy as np
from mathutils import Matrix


def keep_top_k_components_by_face_count(ms: ml.MeshSet, k: int = 2):
    """
    Keeps the top-k largest connected components (by face count) in a MeshSet.
    Operates in-place on the provided MeshSet.
    """
    print(f"Splitting mesh into connected components...")
    #original_index = ms.current_mesh_index()

    # Split into layers without deleting original
    ms.apply_filter('generate_splitting_by_connected_components', delete_source_mesh=False)

    num_components = ms.number_meshes()
    print(f"Total layers after split: {num_components}")

    if num_components <= k:
        print(f"{num_components} component(s) found, which is <= {k}. Nothing will be deleted.")
        return

    # Step 1: Gather face counts
    component_info = []
    for i in range(num_components):
        mesh = ms.mesh(i)
        face_count = mesh.face_number()
        component_info.append((i, face_count))
        print(f"Component {i}: {face_count} faces")

    # Step 2: Sort and select top-k by face count
    component_info.sort(key=lambda x: x[1], reverse=True)
    keep_indices = set(idx for idx, _ in component_info[:k])
    print(f"Keeping top {k} components: {sorted(keep_indices)}")

    # Step 3: Delete all others
    for i in reversed(range(num_components)):
        if i not in keep_indices:
            ms.set_current_mesh(i)
            ms.delete_current_mesh()

    # Optional: reset to first kept mesh
    ms.set_current_mesh(0)
    ms.delete_current_mesh()
    print(f"Remaining meshes: {ms.mesh_number()} (top {k} components retained)")


def clean_dense_mesh(
    input_mesh: Path,
    output_mesh: Path,
    min_component_diag: float = 0.2,
    distance_percentile: float = 95.0, # Percentile of distances for radius
    keep_largest_component_after_trim: bool = True, # NEW: Option to apply this final step
    recompute_normals: bool = True,
    k_neighbors: int = 12,
    remove_duplicates: bool = True,
    remove_unref: bool = True,
    remove_zero_area: bool = True,
    binary: bool = True,
):
    ms = ml.MeshSet()
    ms.load_new_mesh(str(input_mesh))
    print(f"Loaded mesh: {input_mesh}")
    print(f"Initial mesh has {ms.current_mesh().vertex_number()} vertices and {ms.current_mesh().face_number()} faces.")

    if remove_duplicates:
        print("Removing duplicate vertices...")
        ms.meshing_remove_duplicate_vertices()
        print(f"After duplicates: {ms.current_mesh().vertex_number()} vertices.")

    if remove_unref:
        print("Removing unreferenced vertices...")
        ms.meshing_remove_unreferenced_vertices()
        print(f"After unreferenced: {ms.current_mesh().vertex_number()} vertices.")

    # First pass of connected component removal (by diameter)
    # This removes small *disconnected* dust/noise from the original mesh.
    print(f"Removing initial small connected components by diameter (min_component_diag: {min_component_diag * 100:.2f}%)...")
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=ml.PercentageValue(min_component_diag * 100.0),
        removeunref=True
    )
    print(f"After initial component removal: {ms.current_mesh().vertex_number()} vertices and {ms.current_mesh().face_number()} faces.")

    # --- Logic: Calculate Centroid and Robust Radius ---
    current_mesh = ms.current_mesh()
    if current_mesh.vertex_number() == 0:
        print("No vertices left after initial cleaning, cannot determine radius.")
        ms.save_current_mesh(str(output_mesh), binary=binary)
        return

    print("Calculating centroid and vertex distances...")

    # Get the bounding box object from the mesh, and then get the center from the bounding box.
    bbox = current_mesh.bounding_box()
    centroid = bbox.center()

    vertices = current_mesh.vertex_matrix() # Get all vertex coordinates as a NumPy array

    if vertices.shape[0] == 0:
        print("No vertices found after cleaning. Exiting.")
        ms.save_current_mesh(str(output_mesh), binary=binary)
        return

    # Calculate distances of all vertices from the centroid
    distances = np.linalg.norm(vertices - centroid, axis=1)

    # Determine sphere_trim_radius based on a percentile of these distances
    sphere_trim_radius = np.percentile(distances, distance_percentile)

    print(f"Calculated centroid (Bounding Box Center): {centroid}")
    print(f"Max distance from centroid: {np.max(distances):.3f}")
    print(f"Auto-calculated sphere_trim_radius: {sphere_trim_radius:.3f} (using {distance_percentile} percentile)")

    # Apply spherical trim
    if sphere_trim_radius > 0:
        print("Applying spherical trim...")
        cx, cy, cz = centroid[0], centroid[1], centroid[2]
        r = sphere_trim_radius

        # Use explicit multiplication for squaring
        condition_string = (
            f"((x - {cx}) * (x - {cx})) + "
            f"((y - {cy}) * (y - {cy})) + "
            f"((z - {cz}) * (z - {cz})) > ({r} * {r})"
        )

        ms.apply_filter('compute_selection_by_condition_per_vertex',
                        condselect=condition_string)

        ms.apply_filter('meshing_remove_selected_vertices')
        ms.apply_filter('meshing_remove_unreferenced_vertices')
        print(f"After spherical trim: {ms.current_mesh().vertex_number()} vertices and {ms.current_mesh().face_number()} faces.")
    # --- End Logic ---

    if keep_largest_component_after_trim:
        print("Keeping top 2 largest connected components...")
        keep_top_k_components_by_face_count(ms, k=2)
    
    if recompute_normals:
        print("Recomputing normals...")
        ms.compute_normal_for_point_clouds(
            k=k_neighbors,
            smoothiter=0,
            flipflag=False,
            viewpos=[0,0,0]
        )
        print("Normals recomputed.")

    ms.save_current_mesh(str(output_mesh), binary=binary)
    print(f"Final mesh saved to: {output_mesh}")


def center_and_rotate_ply(input_ply_file, output_ply_file, rotation=None):
    """
    Load input_ply_file, translate the centroid to (0,0,0),
    optionally rotate by RX,RY,RZ degrees (Euler), and write to output_ply_file.
    
    :param input_ply_file: path to input PLY file
    :param output_ply_file: path to output PLY file
    :param rotation: dict with optional 'RX', 'RY', 'RZ' in degrees
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(input_ply_file)
    
    # Compute centroid and translate to origin
    centroid = pcd.get_center()
    pcd.translate(-centroid)
    
    # If rotation is specified, apply in order RX -> RY -> RZ
    if rotation is not None:
        rx = np.deg2rad(rotation.get('RX', 0))
        ry = np.deg2rad(rotation.get('RY', 0))
        rz = np.deg2rad(rotation.get('RZ', 0))
        
        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        pcd.rotate(R)
    
    # Write out the transformed point cloud
    o3d.io.write_point_cloud(output_ply_file, pcd)


def center_txt_points3d(input_txt_file, output_txt_file, rotation=None):
    """
    Load COLMAP points3D.txt, center its points at (0,0,0),
    optionally rotate by RX, RY, RZ in degrees, and write to output_txt_file.
    """
    import numpy as np

    # Parse the input file
    points = []
    lines = []
    with open(input_txt_file, 'r') as f:
        for line in f:
            lines.append(line)
            if line.startswith("#") or line.strip() == "":
                continue
            tokens = line.split()
            x, y, z = map(float, tokens[1:4])
            points.append([x, y, z])

    points = np.array(points)
    centroid = points.mean(axis=0)
    print(f"Centroid: {centroid}")

    # Prepare rotation if provided
    if rotation is not None:
        rx, ry, rz = np.deg2rad(rotation.get("RX", 0)), np.deg2rad(rotation.get("RY", 0)), np.deg2rad(rotation.get("RZ", 0))
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
    else:
        R = np.identity(3)

    # Now rewrite lines, shifting + rotating the points
    with open(output_txt_file, 'w') as fout:
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                fout.write(line)
                continue
            tokens = line.split()
            x, y, z = map(float, tokens[1:4])
            p = np.array([x, y, z]) - centroid
            p_rot = R @ p
            tokens[1:4] = [f"{p_rot[0]}", f"{p_rot[1]}", f"{p_rot[2]}"]
            fout.write(' '.join(tokens) + '\n')

    print(f"Written centered points to {output_txt_file}")


def transform_point_cloudxx(input_file, output_file, scale=1.0, rotation=None, translation=None):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file)
    
    if scale != 1.0:
        ms.apply_filter('transform_scale_normalize', scaleabsolute=scale)
    
    if rotation:
        if rotation.get('rx', 0.0):
            ms.apply_filter('transform_rotate_centered', axisx=1, axisy=0, axisz=0, angle=rotation['rx'])
        if rotation.get('ry', 0.0):
            ms.apply_filter('transform_rotate_centered', axisx=0, axisy=1, axisz=0, angle=rotation['ry'])
        if rotation.get('rz', 0.0):
            ms.apply_filter('transform_rotate_centered', axisx=0, axisy=0, axisz=1, angle=rotation['rz'])

    if translation:
        if any(translation.get(axis, 0.0) != 0.0 for axis in ['tx', 'ty', 'tz']):
            ms.apply_filter(
                'transform_translate',
                axisx=translation.get('tx', 0.0),
                axisy=translation.get('ty', 0.0),
                axisz=translation.get('tz', 0.0)
            )
    
    ms.save_current_mesh(output_file)
  

def transform_point_cloud(input_file, output_file, scale=1.0, rotation=None, translation=None):
    """
    Applies scale, rotation, and translation to a point cloud using a single
    transformation matrix, and writes the output using Open3D.
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(input_file)

    # Initialize a 4x4 identity matrix.
    transform_matrix = np.identity(4)

    # 1. --- Create Scaling Matrix ---
    if scale != 1.0:
        scale_matrix = np.array([
            [scale, 0,     0,     0],
            [0,     scale, 0,     0],
            [0,     0,     scale, 0],
            [0,     0,     0,     1]
        ])
        transform_matrix = scale_matrix @ transform_matrix

    # 2. --- Create Rotation Matrix ---
    if rotation:
        rx_deg = rotation.get('rx', 0.0)
        ry_deg = rotation.get('ry', 0.0)
        rz_deg = rotation.get('rz', 0.0)

        rx_rad, ry_rad, rz_rad = np.radians([rx_deg, ry_deg, rz_deg])

        cx, sx = np.cos(rx_rad), np.sin(rx_rad)
        cy, sy = np.cos(ry_rad), np.sin(ry_rad)
        cz, sz = np.cos(rz_rad), np.sin(rz_rad)

        rot_x_matrix = np.array([[1,0,0,0], [0,cx,-sx,0], [0,sx,cx,0], [0,0,0,1]])
        rot_y_matrix = np.array([[cy,0,sy,0], [0,1,0,0], [-sy,0,cy,0], [0,0,0,1]])
        rot_z_matrix = np.array([[cz,-sz,0,0], [sz,cz,0,0], [0,0,1,0], [0,0,0,1]])

        rotation_matrix = rot_z_matrix @ rot_y_matrix @ rot_x_matrix
        transform_matrix = rotation_matrix @ transform_matrix

    # 3. --- Create Translation Matrix ---
    if translation:
        tx = translation.get('tx', 0.0)
        ty = translation.get('ty', 0.0)
        tz = translation.get('tz', 0.0)

        translation_matrix = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        transform_matrix = translation_matrix @ transform_matrix

    # 4. --- Apply the transformation ---
    if not np.array_equal(transform_matrix, np.identity(4)):
        pcd.transform(transform_matrix)

    # 5. --- Save the result ---
    o3d.io.write_point_cloud(output_file, pcd)


def build_transform_matrix(scale, rotation, translation):
    transform_matrix = np.identity(4)

    # Scale
    if scale != 1.0:
        scale_matrix = np.array([
            [scale,0,0,0],
            [0,scale,0,0],
            [0,0,scale,0],
            [0,0,0,1]
        ])
        transform_matrix = scale_matrix @ transform_matrix

    # Rotation
    rx, ry, rz = np.radians([rotation.get('rx',0.0), rotation.get('ry',0.0), rotation.get('rz',0.0)])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]])
    rot_y = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]])
    rot_z = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]])
    rotation_matrix = rot_z @ rot_y @ rot_x
    transform_matrix = rotation_matrix @ transform_matrix

    # Translation
    tx, ty, tz = translation.get('tx',0.0), translation.get('ty',0.0), translation.get('tz',0.0)
    translation_matrix = np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]])
    transform_matrix = translation_matrix @ transform_matrix

    return transform_matrix



def transform_points3D_txt(input_file, output_file, transform_matrix):
    """
    Transforms a COLMAP points3D.txt file using the given 4x4 transform matrix
    (applies scaling, rotation, and translation).
    """
    if np.allclose(transform_matrix, np.identity(4)):
        print("Transform matrix is identity. Copying without change.")
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in fin:
                fout.write(line)
        return

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if line.startswith('#') or line.strip() == '':
                fout.write(line)
                continue
            tokens = line.strip().split()
            xyz = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), 1.0])
            xyz_trans = transform_matrix @ xyz
            tokens[1], tokens[2], tokens[3] = f"{xyz_trans[0]:.8f}", f"{xyz_trans[1]:.8f}", f"{xyz_trans[2]:.8f}"
            fout.write(" ".join(tokens) + "\n")
    print(f"Transformed points3D.txt written to {output_file}")

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,     1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w,     2*y*z+2*x*w,   1-2*x**2-2*y**2]
    ])

def rotmat2qvec(R):
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def transform_images_txt(input_file, output_file, transform_matrix):
    R_world = transform_matrix[:3,:3]
    t_world = transform_matrix[:3,3]

    if np.allclose(transform_matrix, np.identity(4)):
        print("Transform matrix is identity. Copying without change.")
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in fin:
                fout.write(line)
        return

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if line.startswith('#') or line.strip() == '':
                fout.write(line)
                continue
            tokens = line.strip().split()
            qvec = np.array(list(map(float, tokens[1:5])))
            tvec = np.array(list(map(float, tokens[5:8])))

            # Original camera rotation and camera center
            R_cw = qvec2rotmat(qvec)
            C = - R_cw.T @ tvec

            # Apply world transform to camera center
            C_trans = R_world @ C + t_world

            # Adjust camera rotation because world rotated
            R_cw_trans = R_cw @ R_world.T
            qvec_trans = rotmat2qvec(R_cw_trans)

            # Recompute tvec for new camera pose
            tvec_trans = - R_cw_trans @ C_trans

            new_line = "{} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {} {}\n".format(
                tokens[0],
                *qvec_trans,
                *tvec_trans,
                tokens[8], tokens[9]
            )
            fout.write(new_line)
    print(f"Transformed images.txt written to {output_file}")


def compute_blender_camera_transform(images_file):
    """
    Given a COLMAP images.txt, compute the Blender camera transform
    (location XYZ and rotation quaternion WXYZ), adjusted for coordinate system.
    """

    import numpy as np
    from mathutils import Matrix

    def qvec2rotmat(qvec):
        w, x, y, z = qvec
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w,   2*x*z+2*y*w],
            [2*x*y+2*z*w,     1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w,     2*y*z+2*x*w,   1-2*x**2-2*y**2]
        ])

    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            tokens = line.strip().split()
            qvec = np.array(list(map(float, tokens[1:5])))
            tvec = np.array(list(map(float, tokens[5:8])))

            # Compute camera center
            R_cw = qvec2rotmat(qvec)
            C = - R_cw.T @ tvec  # camera center in COLMAP world

            # Compute Blender rotation matrix (camera-to-world)
            R_wc = R_cw.T

            # === Adjust for COLMAP -> Blender axes ===
            # COLMAP: X right, Y down, Z forward
            # Blender: X right, Y forward, Z up
            M = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])

            C_blender = M @ C
            R_wc_blender = M @ R_wc

            # Convert to Blender mathutils for quaternion
            R_blender = Matrix((R_wc_blender[0], R_wc_blender[1], R_wc_blender[2])).transposed()
            quat = R_blender.to_quaternion()

            print("\n🎯 Blender adjusted camera transform:")
            print(f"Adjusted location XYZ: ({C_blender[0]:.6f}, {C_blender[1]:.6f}, {C_blender[2]:.6f})")
            print(f"Adjusted rotation quaternion WXYZ: ({quat.w:.6f}, {quat.x:.6f}, {quat.y:.6f}, {quat.z:.6f})")
            return  # only process first image
        
        
def add_axes(filename):
    """
    Adds colored XYZ axes points to a PLY point cloud file.
    - White at (0,0,0)
    - Red at (1,0,0)
    - Green at (0,1,0)
    - Blue at (0,0,1)
    Writes output to <original_name>_with_axes.ply
    """
    pcd = o3d.io.read_point_cloud(filename)

    # Make sure we have colors
    if len(pcd.colors) == 0:
        colors = np.ones((len(pcd.points), 3)) * 0.5  # gray default
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Axis points and colors
    axis_points = np.array([
        [0,0,0],  # origin
        [1,0,0],  # +X
        [0,1,0],  # +Y
        [0,0,1]   # +Z
    ])
    axis_colors = np.array([
        [1,1,1],  # white
        [1,0,0],  # red
        [0,1,0],  # green
        [0,0,1]   # blue
    ])

    # Extend existing point cloud
    combined_points = np.vstack((np.asarray(pcd.points), axis_points))
    combined_colors = np.vstack((np.asarray(pcd.colors), axis_colors))

    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    # Save new file
    out_file = Path(filename).with_stem(Path(filename).stem + "_with_axes")
    o3d.io.write_point_cloud(str(out_file), pcd)
    print(f"✅ Saved: {out_file}")
