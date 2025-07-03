import os
import re
import sys
import json
import time
import pandas as pd
import subprocess
from pathlib import Path


from scripts.utils import run_subprocess, DOCKER_COMPOSE_PREFIX
from loguru import logger
from datetime import datetime

def run_colmap_feature_extractor(image_path, db_path):
    run_subprocess([
        "colmap",
        "colmap",
        "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_path,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",          # assume pinhole, can be changed if needed
        "--SiftExtraction.use_gpu", "0",                   # CPU for stability
        "--SiftExtraction.num_threads", "8",               # 8 is good starting point
        "--SiftExtraction.estimate_affine_shape", "0",     # faster, fine for drone
        "--SiftExtraction.domain_size_pooling", "0",       # faster, fine for drone
        "--SiftExtraction.max_image_size", "3200",         # important: limit to ~3000-3500px max
    ], "COLMAP FeatureExtractor")

def run_colmap_exhaustive_matcher(db_path):
    run_subprocess([
        "colmap",
        "colmap",
        "exhaustive_matcher",
        "--database_path", db_path,
        "--SiftMatching.use_gpu", "0",          # 🚀 CRUCIAL
        "--SiftMatching.num_threads", "8"  # safe limit
    ], "COLMAP ExhaustiveMatcher")

def run_colmap_sequential_matcher(db_path):
    run_subprocess([
        "colmap",
        "colmap",
        "sequential_matcher",
        "--database_path", db_path,
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", "8",
        "--SequentialMatching.overlap", "5",  # Match to 5 neighbors forward and backward
    ], "COLMAP SequentialMatcher")

def run_colmap_sequential_matcher2(db_path):
    run_subprocess([
        "colmap",
        "colmap",
        "sequential_matcher",
        "--database_path", db_path,
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", "8",
        "--SequentialMatching.overlap", "5",         # Tune this for your FPS
        "--SequentialMatching.quadratic_overlap", "0", # Don't scale overlap with time
        "--SequentialMatching.loop_detection", "0",    # Optional: disable loop detection for linear flight
    ], "COLMAP SequentialMatcher")

def run_colmap_mapper(db_path, image_path, output_path):
    run_subprocess([
        "colmap",
        "colmap",
        "mapper",
        "--database_path", db_path,
        "--image_path", image_path,
        "--output_path", output_path,
        "--Mapper.num_threads", "8"  # safe limit
    ], "COLMAP Mapper (Sparse Reconstruction)")

def run_colmap_model_converter(input_model_path, output_ply_path):
    run_subprocess([
        "colmap",
        "colmap",
        "model_converter",
        "--input_path", input_model_path,
        "--output_path", output_ply_path,
        "--output_type", "PLY"
    ], "COLMAP ModelConverter (Export PLY)")

def run_colmap_model_analyzer(model_folder_in_container, stats_file, elapsed_time):
    """
    Run COLMAP model_analyzer inside container and save output as JSON in colmap/stats/model_analyzer.json.
    `model_folder_in_container` should be like /projects/project_name/colmap/sparse/0
    """
    model_path = Path(model_folder_in_container)
    
    # Extract scenario name and model_id from container path
    scenario_name = model_path.parts[2]
    model_id = model_path.parts[4]

    stats_file = Path(stats_file)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model folder in container: {model_folder_in_container}")
    logger.info(f"Scenario name: {scenario_name}")
    logger.info(f"Model id: {model_id}")

    cmd = DOCKER_COMPOSE_PREFIX + [
        "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "colmap", "colmap", "model_analyzer",
        "--path", model_folder_in_container
    ]

    logger.info(f"📊 Running COLMAP model_analyzer for: {scenario_name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stderr

        video, config = scenario_name.split("-", 1)
        format_, fps, max_dim, filter_ = config.split("_")

        stats = {
            "Scenario": scenario_name,
            "Model ID": model_id,
            "Video": video,
            "Format": format_,
            "Filter": filter_,
            "FPS": float(fps),
            "MaxDim": int(max_dim),
            "Timestamp": datetime.now().isoformat(),
            "Elapsed": elapsed_time,
        }

        patterns = {
            "Cameras": r"\]\s+Cameras:\s+(\d+)",
            "Images": r"\]\s+Images:\s+(\d+)",
            "Registered Images": r"\]\s+Registered images:\s+(\d+)",
            "Points3D": r"\]\s+Points:\s+(\d+)",
            "Observations": r"\]\s+Observations:\s+(\d+)",
            "Mean Track Length": r"\]\s+Mean track length:\s+([\d\.]+)",
            "Mean Observations per Image": r"\]\s+Mean observations per image:\s+([\d\.]+)",
            "Mean Reprojection Error": r"\]\s+Mean reprojection error:\s+([\d\.]+)"
        }

        for key, pattern in patterns.items():
            stats[key] = "?"
            for line in output.splitlines():
                match = re.search(pattern, line)
                if match:
                    stats[key] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                    break

        df = pd.DataFrame([stats])

        try:
            df["pts_per_img"] = df["Points3D"] / df["Images"]
            df["obs_per_pt"] = df["Observations"] / df["Points3D"]
            df["obs_per_img"] = df["Observations"] / df["Images"]
            df["quality"] = df["Points3D"] / df["Mean Reprojection Error"]
            df["points_per_registered_img"] = df["Points3D"] / df["Registered Images"]
            df["obs_per_registered_img"] = df["Observations"] / df["Registered Images"]
            df["obs_per_cam"] = df["Observations"] / df["Cameras"]
            df["pts_per_cam"] = df["Points3D"] / df["Cameras"]
        except ZeroDivisionError as e:
            logger.warning(f"⚠️ Division by zero in derived metrics: {e}")

        df_out = df.to_dict(orient="records")[0]
        stats_file.write_text(json.dumps(df_out, indent=2))

        txt_filename = stats_file.with_suffix(".txt")
        txt_filename.write_text(output)

        logger.success(f"✅ ModelAnalyzer stats saved → {stats_file}")
        return stats

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ model_analyzer failed for {scenario_name}: {e.output}")
        raise


def host_to_container_path(host_path):
    if not os.path.abspath(host_path).startswith(os.path.abspath("projects")):
        raise ValueError(f"Path {host_path} is outside of projects/ folder!")
    return "/projects/" + os.path.relpath(host_path, "projects")

from pathlib import Path
import shutil
from loguru import logger

def flatten_sparse_model(sparse_root: str | Path, model_id: str = "0"):
    """
    Move the contents of sparse_root/<model_id>/ up to sparse_root/, then delete the nested model_id folder.

    Parameters:
    - sparse_root (str or Path): Path to the sparse/ folder (e.g. "projects/.../colmap/0/sparse")
    - model_id (str): Subfolder name to flatten (usually "0")
    """
    sparse_root = Path(sparse_root)
    model_path = sparse_root / model_id

    if not model_path.exists():
        logger.warning(f"⚠️ Model path does not exist: {model_path}")
        return

    logger.info(f"📁 Flattening model {model_id} → {sparse_root}")

    for file in model_path.iterdir():
        target = sparse_root / file.name
        if target.exists():
            logger.warning(f"⚠️ Skipping existing file: {target}")
            continue
        shutil.move(str(file), str(target))
        logger.debug(f"Moved {file} → {target}")

    # Clean up empty folder
    try:
        model_path.rmdir()
        logger.success(f"✅ Flattened model '{model_id}' and removed folder.")
    except OSError as e:
        logger.error(f"❌ Failed to remove {model_path}: {e}")


def run_colmap_pipeline(image_path, colmap_output_folder):
    """Create and run the COLMAP pipeline using Path objects."""
    
    start_time = time.time()  # ⏱️ Start timer
    
    # 1️⃣ Ensure host folders exist
    colmap_output_folder = Path(colmap_output_folder)
    sparse_folder = colmap_output_folder / "sparse"
    stats_folder = colmap_output_folder / "stats"
    stats_file = stats_folder / f"model_analyzer-{colmap_output_folder.name}.json"

    sparse_folder.mkdir(parents=True, exist_ok=True)
    stats_folder.mkdir(parents=True, exist_ok=True)

    # 2️⃣ Compute host paths 
    db_path_host = colmap_output_folder / "db.db"

    # 3️⃣ Map to container paths
    image_path_in_container = host_to_container_path(str(image_path))
    db_path_in_container = host_to_container_path(str(db_path_host))
    sparse_folder_in_container = host_to_container_path(str(sparse_folder))
    
    model_0_folder_host = sparse_folder
    model_0_folder_in_container = f"{sparse_folder_in_container}" 

    ply_output_path_host = str( sparse_folder / "0.ply") 
    ply_output_path_in_container = str( Path(sparse_folder_in_container) / "0.ply" )

    logger.info(f"📦 image_path_in_container      : {image_path_in_container}")
    logger.info(f"📂 db_path_host                 : {db_path_host}")
    logger.info(f"📦 db_path_in_container         : {db_path_in_container}")
    logger.info(f"📦 sparse_folder_in_container   : {sparse_folder_in_container}")
    logger.info(f"📂 model_0_folder_host          : {model_0_folder_host}")
    logger.info(f"📦 model_0_folder_in_container  : {model_0_folder_in_container}")
    logger.info(f"📂 ply_output_path_host         : {ply_output_path_host}")
    logger.info(f"📦 ply_output_path_in_container : {ply_output_path_in_container}")

    if 1:
        # 4️⃣ Run pipeline steps with CONTAINER paths
        run_colmap_feature_extractor(image_path_in_container, db_path_in_container)
        run_colmap_sequential_matcher(db_path_in_container)
        run_colmap_mapper(db_path_in_container, image_path_in_container, sparse_folder_in_container)

        # 4.5️⃣ Flatten sparse/0 → sparse/
        flatten_sparse_model(sparse_folder, model_id="0")

    # 5️⃣ Check if model was produced
    points3D_bin_host = model_0_folder_host / "points3D.bin"
    if points3D_bin_host.exists():
        logger.info(f"✅ Mapper produced model — exporting PLY to {ply_output_path_host}")
        run_colmap_model_converter(model_0_folder_in_container, ply_output_path_in_container)

        elapsed_time = time.time() - start_time
        run_colmap_model_analyzer(model_0_folder_in_container, str(stats_file), elapsed_time)
        # 6️⃣ Count points (optional)
        from scripts.utils import count_ply_points
        num_points = count_ply_points(ply_output_path_host)
        logger.info(f"📈 COLMAP sparse model contains {num_points} points.")
    else:
        logger.warning("⚠️ Mapper did not produce a model — skipping PLY export.")

from pathlib import Path

def run_colmap_point_filtering(input_model_host, output_model_host, min_track_len=2, max_reproj_error=4.0, min_tri_angle=1.5):
    """Run COLMAP point_filtering inside a Docker container with project-relative paths."""

    input_model_host = Path(input_model_host)
    output_model_host = Path(output_model_host)

    # Verify required file
    points3d_bin = input_model_host / "points3D.bin"
    if not points3d_bin.exists():
        logger.error(f"❌ points3D.bin not found in {input_model_host}")
        return

    output_model_host.mkdir(parents=True, exist_ok=True)

    # Map to container paths
    input_model_container = host_to_container_path(str(input_model_host))
    output_model_container = host_to_container_path(str(output_model_host))

    ply_output_path_in_container = str(output_model_host.parent / f"{output_model_host.name}.ply")
    ply_output_path_in_container = host_to_container_path(ply_output_path_in_container)

    run_subprocess([
        "colmap",
        "colmap",
        "point_filtering",
        "--input_path", input_model_container,
        "--output_path", output_model_container,
        "--min_track_len", str(min_track_len),
        "--max_reproj_error", str(max_reproj_error),
        "--min_tri_angle", str(min_tri_angle)
    ], "COLMAP Model Cleaner")

    run_colmap_model_converter(output_model_container, ply_output_path_in_container)

    stats_folder = output_model_host.parents[1] / "stats"
    stats_file = stats_folder / f"model_analyzer-{output_model_host.parents[1].name}.json"
    run_colmap_model_analyzer(output_model_container, str(stats_file), 0.0)
