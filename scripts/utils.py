import os
import sys
import subprocess
import time
import json
import re
import matplotlib.pyplot as plt

from loguru import logger

# Configure Loguru
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")
logger.add("logs/pipeline.log", level="DEBUG", format="{time} | {level} | {message}")

# Always point explicitly to the docker-compose.yml
DOCKER_COMPOSE_PREFIX = [
    "docker", "compose", "-f", "./docker/docker-compose.yml"
]

def run_subprocess(cmd_suffix, step_name):
    """Run a subprocess with live console output and log capture."""
    full_cmd = DOCKER_COMPOSE_PREFIX + [
        "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}"
    ] + cmd_suffix

    logger.info(f"👉 Running [{step_name}]: {' '.join(full_cmd)}")

    start_time = time.time()

    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        with process.stdout:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"[{step_name}] {line}")

        returncode = process.wait()
        duration = time.time() - start_time

        if returncode == 0:
            logger.success(f"✅ Step succeeded: {step_name} (Elapsed time: {duration:.1f} sec)")
        else:
            logger.error(f"❌ Error during step: {step_name} (Exit code {returncode})")
            logger.error(f"❌ Aborting pipeline after failed step: {step_name}")
            sys.exit(returncode)

    except Exception as e:
        logger.exception(f"❌ Exception during step: {step_name}: {str(e)}")
        sys.exit(1)


def validate_sfm_data_paths(sfm_engine):
    """Validate that all image paths in sfm_data.json are relative."""
    
    sfm_json_path = "data/openmvg/reconstruction_incremental/sfm_data.json" if sfm_engine == "INCREMENTAL" \
        else "data/openmvg/reconstruction_global/sfm_data.json"

    with open(sfm_json_path) as f:
        data = json.load(f)

    views = data.get("views", [])
    bad_paths = []

    for view in views:
        path = view["value"]["ptr_wrapper"]["data"]["filename"]
        if path.startswith("/"):
            bad_paths.append(path)

    if bad_paths:
        logger.error("⚠️ WARNING: Some image paths are absolute! You may hit OpenMVS errors.")
        for p in bad_paths[:5]:
            logger.error(f"  - {p}")
        logger.error("👉 Fix this by using '--use_relative_path' in SfMInit_ImageListing.")
        logger.error("❌ Aborting pipeline to prevent wasting time.")
        sys.exit(1)
    else:
        logger.info("✅ All image paths in sfm_data.json are relative → OK.")


def check_and_select_best_matches(match_files, min_valid_matches=10, min_matches_per_pair=10):
    """Check match files and select the best one."""
    logger.info("🔍 Checking match statistics to select best match file...")

    selected_file = None
    best_valid_pairs = -1

    for match_file in match_files:
        if not os.path.exists(match_file):
            logger.warning(f"⚠️ Match file not found: {match_file}")
            continue

        valid_pairs = 0
        with open(match_file) as f:
            content = f.read()

        pairs = re.findall(r'(?m)^(\d+) (\d+)\n(\d+)', content)

        for pair in pairs:
            num_matches = int(pair[2])
            if num_matches >= min_matches_per_pair:
                valid_pairs += 1

        logger.info(f"ℹ️ [{match_file}] → {valid_pairs} valid pairs (≥{min_matches_per_pair} matches)")

        if valid_pairs >= min_valid_matches and valid_pairs > best_valid_pairs:
            best_valid_pairs = valid_pairs
            selected_file = match_file

    if selected_file is None:
        logger.error(f"❌ No match file passed the threshold of {min_valid_matches} valid pairs.")
        logger.error("❌ Aborting pipeline prior to SfM.")
        sys.exit(1)

    logger.success(f"✅ Selected match file: {selected_file} with {best_valid_pairs} valid pairs")
    return selected_file

def count_ply_points(ply_path):
    """Count number of points in a PLY file (handles binary or ascii PLY)."""
    if not os.path.exists(ply_path):
        logger.error(f"❌ PLY file not found: {ply_path}")
        return 0

    try:
        with open(ply_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline()
                header_lines.append(line)
                if line.strip() == b"end_header":
                    break

        # Decode header lines as ASCII
        header_lines_decoded = [line.decode("ascii", errors="ignore") for line in header_lines]

        # Search for 'element vertex N'
        num_points = 0
        for line in header_lines_decoded:
            if line.startswith("element vertex"):
                num_points = int(line.split()[-1])
                break

        return num_points

    except Exception as e:
        logger.error(f"❌ Error reading PLY file {ply_path}: {e}")
        return 0


def plot_match_histogram(match_file, out_path):
    """Plot a histogram of number of matches per image pair."""
    if not os.path.exists(match_file):
        logger.error(f"❌ Match file not found: {match_file}")
        return

    match_counts = []

    with open(match_file) as f:
        content = f.read()

    pairs = re.findall(r'(?m)^(\d+) (\d+)\n(\d+)', content)

    for pair in pairs:
        num_matches = int(pair[2])
        match_counts.append(num_matches)

    if not match_counts:
        logger.warning(f"⚠️ No match pairs found in: {match_file}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(match_counts, bins=50, color="skyblue", edgecolor="black")
    plt.title(f"Match Count Histogram\n{os.path.basename(match_file)}")
    plt.xlabel("Number of Matches per Image Pair")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.success(f"✅ Saved match histogram → {out_path}")

def verify_minimum_poses( minimum_poses, sfm_engine ):
    # Check number of views
    recon_json_path = "data/openmvg/reconstruction_incremental/sfm_data.json" if sfm_engine == "INCREMENTAL" \
        else "data/openmvg/reconstruction_global/sfm_data.json"

    with open(recon_json_path) as f:
        data = json.load(f)

    num_poses = len(data.get("views", {}))
    logger.info(f"Number of views (poses) reconstructed: {num_poses}")

    MIN_POSES = minimum_poses
    if num_poses < MIN_POSES:
        logger.error(f"❌ Too few poses ({num_poses}) — aborting pipeline.")
        sys.exit(1)
