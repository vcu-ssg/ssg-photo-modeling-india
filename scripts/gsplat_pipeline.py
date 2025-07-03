import os
import sys
from scripts.utils import run_subprocess
from loguru import logger
from pathlib import Path

def host_to_container_path(host_path: str) -> str:
    """Convert a host path like projects/project_name/xxx/... to container path /projects/project_name/xxx """
    abs_host = os.path.abspath(host_path)
    abs_data_root = os.path.abspath("projects")

    if not abs_host.startswith(abs_data_root):
        raise ValueError(f"Path {host_path} must be within the project ./data directory.")

    rel_path = os.path.relpath(abs_host, abs_data_root)
    return os.path.join("/projects", rel_path)

def run_ply_to_splat_converter( input_file, output_file ):
    """  """
    logger.info(f"📸 Input PLY file    : {input_file}")
    logger.info(f"📈 Output SPLAT file : {output_file}")

    cmd = [
        "gsplat",
        "python", "/opt/point-cloud-tools/convert.py",
        input_file,
        output_file
    ]

#    cmd = ["gsplat","ls","/opt/point_cloud_tools"]
    run_subprocess(cmd, f"gsplat converter")



def run_gsplat_pipeline(scene, images_dir, sparse_dir, model_dir, iterations=30000, sh_degree=3):
    """ gaussian splatting training pipeline """
    
    logger.info(f"🟢 Running gsplat for {scene}")
    logger.info(f"📸 Host Images      : {images_dir}")
    logger.info(f"📈 Host Sparse model: {sparse_dir}")
    logger.info(f"💾 Host Output/model path : {model_dir}")

    # Validate and create host-side output directory
    try:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs( model_dir + "/point_cloud" )
        logger.success(f"📁 Ensured output directory: {model_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to create output directory {model_dir}: {e}")
        sys.exit(1)

    # Convert to container paths
    frames_container = host_to_container_path(images_dir)
    sparse_container = host_to_container_path(sparse_dir)
    output_container = host_to_container_path(model_dir)

    logger.info(f"📸 Container Images      : {frames_container}")
    logger.info(f"📈 Container Sparse model: {sparse_container}")
    logger.info(f"💾 Container Output path : {output_container}")


# https://medium.com/data-science/turn-yourself-into-a-3d-gaussian-splat-3a2bc59a770f
# resolution=1                      # default 1
# sh_degree=3                       # default 3
# position_lr_init=0.00016          # default = 0.00016, large scale -> 0.000016
# scaling_lr=0.005                  # default = 0.005, large scale -> 0.001
# iterations=30000                  # default 30000
# densify_from_iter=500             # default 500
# densify_until_iter=15000          # default 15000
# test_iterations="7000 30000"      # default 7000 30000
# save_iterations="7000 30000"      # default 7000 30000
# data_device=cpu

    cmd = [
        "gsplat",
        "python", "train.py",
        "--source_path", sparse_container,     # ✅ this is the COLMAP project root
        "--model_path", output_container,       # ✅ full path to sparse/0
        "--images", frames_container,
        "--iterations", iterations,
        "--sh_degree", sh_degree
    ]

    run_subprocess(cmd, f"gsplat [{scene}]")

    point_cloud_file = Path(output_container) / f"point_cloud/iteration_{iterations}/point_cloud.ply"
    splat_file = point_cloud_file.with_suffix( ".splat" )
    
    run_ply_to_splat_converter( str(point_cloud_file), str(splat_file) )
