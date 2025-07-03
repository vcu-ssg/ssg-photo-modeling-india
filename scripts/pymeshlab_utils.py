import pymeshlab as ml
from pathlib import Path
import numpy as np


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

if __name__ == "__main__":
    input_file = Path("scene_dense_mesh.ply")
    output_file = Path("scene_dense_mesh_final_cleaned.ply")

    clean_dense_mesh_robust_radius(
        input_mesh=input_file,
        output_mesh=output_file,
        min_component_diag=0.01,
        distance_percentile=95.0, # Adjust as needed
        keep_largest_component_after_trim=True, # Set to False if you want to keep all pieces
        recompute_normals=True
    )