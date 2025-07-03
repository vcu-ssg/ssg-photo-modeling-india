import os
import shutil
from scripts.utils import run_subprocess, count_ply_points
from loguru import logger

# === OPENMVG ===

def run_sfm_init_image_listing():
    run_subprocess([
        "openmvg",
        "openMVG_main_SfMInit_ImageListing",
        "-i", "/data/frames",
        "-d", "/usr/local/lib/openMVG/sensor_width_camera_database.txt",
        "-o", "/data/openmvg",
        "-f", "4.5"
    ], "openMVG_main_SfMInit_ImageListing")


def run_compute_features():
    run_subprocess([
        "openmvg",
        "openMVG_main_ComputeFeatures",
        "-i", "/data/openmvg/sfm_data.json",
        "-o", "/data/openmvg/matches"
    ], "openMVG_main_ComputeFeatures")


def run_compute_matches(matches_ratio,geometric_model,match_file):
    """ """
    models = dict(FUNDAMENTAL="f",ESSENTIAL="e",HOMOGRAPHY="h")
    run_subprocess([
        "openmvg",
        "openMVG_main_ComputeMatches",
        "-i", "/data/openmvg/sfm_data.json",
        "-o", f"/data/openmvg/matches/{match_file}",
        "--ratio", str(matches_ratio),
#        "--geometric_model",f"{models.get(geometric_model,"INVALID MODEL")}",
    ], f"openMVG_main_ComputeMatches ({geometric_model})")


def run_export_matches_visualization(engine, matchfile):
    run_subprocess([
        "openmvg",
        "openMVG_main_exportMatches",
        "-i", "/data/openmvg/sfm_data.json",
        "-d", "/data/openmvg/matches",
        "-o", f"/data/openmvg/matches/matches_visualization_{matchfile.replace('.txt','.svg')}",
        "-m", f"/data/openmvg/matches/{matchfile}"
    ], f"Export Matches Visualization ({matchfile})")

    if 0:
        try:
            svg_path = f"data/openmvg/matches/matches_visualization_{engine}_{matchfile.replace('.txt','.svg')}"
            visuals_dest = "data/visuals"
            os.makedirs(visuals_dest, exist_ok=True)
            shutil.copy(svg_path, visuals_dest)
            logger.success(f"✅ Copied {os.path.basename(svg_path)} → {visuals_dest}")
        except Exception as e:
            logger.error(f"❌ Failed to copy '{os.path.basename(svg_path)}' to data/visuals: {e}")


def run_sfm(engine, match_file):
    recon_path = "/data/openmvg/reconstruction_global" if engine == "GLOBAL" else "/data/openmvg/reconstruction_incremental"
    run_subprocess([
        "openmvg",
        "openMVG_main_SfM",
        "-i", "/data/openmvg/sfm_data.json",
        "-m", "/data/openmvg/matches",
        "-o", recon_path,
        "--sfm_engine", engine,
        "--match_file", os.path.basename(match_file)
    ], f"openMVG_main_SfM ({engine})")


def run_convert_sfm_data_format(engine):
    recon_path = "/data/openmvg/reconstruction_global" if engine == "GLOBAL" else "/data/openmvg/reconstruction_incremental"
    run_subprocess([
        "openmvg",
        "openMVG_main_ConvertSfM_DataFormat",
        "-i", f"{recon_path}/sfm_data.bin",
        "-o", f"{recon_path}/sfm_data.json"
    ], "openMVG_main_ConvertSfM_DataFormat")


def run_openmvg_to_openmvs(engine):
    recon_path = "/data/openmvg/reconstruction_global" if engine == "GLOBAL" else "/data/openmvg/reconstruction_incremental"
    run_subprocess([
        "openmvg",
        "openMVG_main_openMVG2openMVS",
        "-i", f"{recon_path}/sfm_data.bin",
        "-d", "/data/openmvs",
        "-o", "/data/openmvs/scene.mvs"
    ], "openMVG_main_openMVG2openMVS")


def export_sparse_ply(engine):
    recon_path = "/data/openmvg/reconstruction_global" if engine == "GLOBAL" else "/data/openmvg/reconstruction_incremental"

    run_subprocess([
        "openmvg",
        "openMVG_main_ComputeStructureFromKnownPoses",
        "-i", f"{recon_path}/sfm_data.bin",
        "-m", "/data/openmvg/matches",
        "-o", f"{recon_path}/robust.ply"
    ], "Export Sparse Point Cloud (PLY)")

    # Optional: copy robust.ply to data/visuals/sparse.ply for consistency
    local_path = "data/openmvg/reconstruction_global" if engine == "GLOBAL" else "data/openmvg/reconstruction_incremental"

    try:
        visuals_dest = "data/visuals"
        os.makedirs(visuals_dest, exist_ok=True)
        shutil.copy(f"{local_path}/robust.ply", f"{visuals_dest}/sparse.ply")
        logger.success("✅ Copied sparse.ply → data/visuals/sparse.ply")
    except Exception as e:
        logger.error(f"❌ Failed to copy sparse.ply: {e}")


# === OPENMVS ===

def run_link_images():
    run_subprocess([
        "openmvs",
        "bash", "-c",
        "ln -sf /data/frames/*.jpg /data/openmvs/"
    ], "Link images into openmvs folder")


def run_densify_pointcloud():
    run_subprocess([
        "openmvs",
        "DensifyPointCloud",
        "-w", "/data/openmvs",
        "scene.mvs"
    ], "OpenMVS DensifyPointCloud")


def export_dense_ply():
    import os
    import shutil

    dense_ply_path = "data/openmvs/scene_dense.ply"
    visuals_dest = "data/visuals"
    os.makedirs(visuals_dest, exist_ok=True)

    # Check if scene_dense.ply exists (produced by DensifyPointCloud)
    if os.path.exists(dense_ply_path):
        shutil.copy(dense_ply_path, f"{visuals_dest}/scene_dense.ply")
        logger.success("✅ Copied scene_dense.ply → data/visuals/scene_dense.ply")

        # Count points
        num_dense_points = count_ply_points(dense_ply_path)
        logger.info(f"scene_dense.ply contains {num_dense_points} points")
        return num_dense_points
    else:
        logger.warning("⚠️ scene_dense.ply not found — dense point cloud was not generated.")
        return 0


def run_reconstruct_mesh():
    run_subprocess([
        "openmvs",
        "ReconstructMesh",
        "-w", "/data/openmvs",
        "scene_dense.mvs"
    ], "OpenMVS ReconstructMesh")


def export_mesh_ply():
    run_subprocess([
        "openmvs",
        "ExportMesh",
        "-w", "/data/openmvs",
        "scene_dense_mesh.mvs",
        "--output-file", "/data/openmvs/scene_dense_mesh.ply"
    ], "Export Mesh (PLY)")

    # Optional: copy to visuals
    try:
        import shutil
        visuals_dest = "data/visuals"
        os.makedirs(visuals_dest, exist_ok=True)
        shutil.copy("/data/openmvs/scene_dense_mesh.ply", f"{visuals_dest}/scene_dense_mesh.ply")
        logger.success("✅ Copied scene_dense_mesh.ply → data/visuals/scene_dense_mesh.ply")
    except Exception as e:
        logger.error(f"❌ Failed to copy scene_dense_mesh.ply: {e}")

def run_texture_mesh():
    run_subprocess([
        "openmvs",
        "TextureMesh",
        "-w", "/data/openmvs",
        "scene_dense_mesh.mvs"
    ], "OpenMVS TextureMesh")


def export_textured_mesh_ply():
    run_subprocess([
        "openmvs",
        "ExportMesh",
        "-w", "/data/openmvs",
        "scene_dense_mesh_texture.mvs",
        "--output-file", "/data/openmvs/scene_dense_mesh_textured.ply"
    ], "Export Textured Mesh (PLY)")

    # Optional: copy to visuals
    try:
        import shutil
        visuals_dest = "data/visuals"
        os.makedirs(visuals_dest, exist_ok=True)
        shutil.copy("/data/openmvs/scene_dense_mesh_textured.ply", f"{visuals_dest}/scene_dense_mesh_textured.ply")
        logger.success("✅ Copied scene_dense_mesh_textured.ply → data/visuals/scene_dense_mesh_textured.ply")
    except Exception as e:
        logger.error(f"❌ Failed to copy scene_dense_mesh_textured.ply: {e}")
