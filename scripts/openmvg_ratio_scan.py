# openmvg_ratio_scan.py

import os
import sys
import json
import shutil

from loguru import logger

from scripts.utils import (
    validate_sfm_data_paths,
    check_and_select_best_matches,
    plot_match_histogram,
    verify_minimum_poses,
)

from scripts.wrappers import (
    run_sfm_init_image_listing,
    run_compute_features,
    run_compute_matches,
    run_sfm,
    run_convert_sfm_data_format,
    run_openmvg_to_openmvs,
    export_sparse_ply,
)

# --- Settings ---
MATCHES_FOLDER = "data/openmvg/matches"
RECON_FOLDER = "data/openmvg/reconstruction_incremental"
OPENMVS_FOLDER = "data/openmvs"
VISUALS_FOLDER = "data/visuals"

# --- Helpers ---

def count_num_poses(sfm_engine):
    sfm_json_path = f"data/openmvg/reconstruction_{sfm_engine.lower()}/sfm_data.json"
    if not os.path.exists(sfm_json_path):
        logger.error(f"Missing sfm_data.json at {sfm_json_path}")
        return 0
    with open(sfm_json_path, "r") as f:
        sfm_data = json.load(f)
    num_poses = len(sfm_data.get("views", []))
    return num_poses

def clean_interim_folders():
    """Remove all generated folders except data/frames/"""
    logger.info("🧹 Cleaning interim folders ...")
    for folder in [MATCHES_FOLDER, RECON_FOLDER, OPENMVS_FOLDER, VISUALS_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logger.info(f"Deleted {folder}")
    os.makedirs(MATCHES_FOLDER, exist_ok=True)
    os.makedirs(VISUALS_FOLDER, exist_ok=True)

# --- Main ---

def run_ratio_scan(
    ratio_values = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
    sfm_engine = "INCREMENTAL",
    min_poses = 3
):

    logger.info("🚀 Starting RATIO SCAN for OpenMVG pipeline ...")

    geometry_model = 'ESSENTIAL'
    geometry_file_template = "matches.e.ratio_{:.2f}.txt"

    # === Scan Ratios ===
    results = []

    for ratio in ratio_values:
        logger.info(f"🔍 Testing ratio={ratio:.2f}")

        # Clean folders first
        clean_interim_folders()

        # === Initialize ===
        run_sfm_init_image_listing()
        run_compute_features()


        # Compute matches
        match_file = geometry_file_template.format(ratio)
        run_compute_matches(ratio, geometry_model, match_file)

        # Plot histogram
        match_path = f"{MATCHES_FOLDER}/{match_file}"
        plot_match_histogram(
            f"{match_path}",
            f"{VISUALS_FOLDER}/match_histogram_{match_file.replace('.txt','.png')}"
        )

        # Select matches (just one here)
        selected_match_file = check_and_select_best_matches(
            [match_path],
            min_valid_matches=10,
            min_matches_per_pair=10
        )

        # Run SfM
        run_sfm(sfm_engine, selected_match_file)

        # Export sparse ply & convert sfm_data
        export_sparse_ply(sfm_engine)
        run_convert_sfm_data_format(sfm_engine)

        # Validate + count poses
        validate_sfm_data_paths(sfm_engine)
        verify_minimum_poses(min_poses, sfm_engine)

        num_poses = count_num_poses(sfm_engine)
        logger.info(f"✅ ratio={ratio:.2f} → {num_poses} poses")

        # Record result
        results.append( (ratio, num_poses) )

    # === Summary ===
    logger.info("📊 RATIO SCAN RESULTS:")
    for ratio, poses in results:
        logger.info(f"    ratio={ratio:.2f} → {poses} poses")

    # Find best ratio
    best_ratio, best_poses = max(results, key=lambda x: x[1])
    logger.info(f"🏆 Best ratio: {best_ratio:.2f} → {best_poses} poses")

    # Optional: write results.csv
    write_results_csv(results)

    logger.info("✅ RATIO SCAN complete.")


def write_results_csv(results, csv_path="data/ratio_scan_results.csv"):
    """Save ratio scan results to CSV."""
    os.makedirs("data", exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("ratio,num_poses\n")
        for ratio, num_poses in results:
            f.write(f"{ratio},{num_poses}\n")
    logger.info(f"💾 Results written to {csv_path}")


# Entry point
if __name__ == "__main__":
    run_ratio_scan(
        ratio_values = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
        sfm_engine = "INCREMENTAL",
        min_poses = 3
    )
