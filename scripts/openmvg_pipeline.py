# openmvg_pipeline.py

import os
import sys
import json

from loguru import logger

from scripts.utils import (
    validate_sfm_data_paths,
    check_and_select_best_matches,
    count_ply_points,
    plot_match_histogram,
    verify_minimum_poses
)

from scripts.wrappers import (
    run_sfm_init_image_listing,
    run_compute_features,
    run_compute_matches,
    run_export_matches_visualization,
    run_sfm,
    run_convert_sfm_data_format,
    run_openmvg_to_openmvs,
    run_link_images,
    run_densify_pointcloud,
    run_reconstruct_mesh,
    run_texture_mesh,
    export_sparse_ply,
    export_dense_ply,
    export_mesh_ply,
    export_textured_mesh_ply,
)

# Main pipeline function
def run_pipeline(enable_texturing=True, sfm_engine="INCREMENTAL", matches_ratio=0.6, min_poses=3):
    """Run OpenMVG + OpenMVS pipeline"""

    logger.info("🚀 Starting OpenMVG + OpenMVS pipeline ...")

    if 1:
        # === OPENMVG ===

        run_sfm_init_image_listing()
        run_compute_features()
        
        match_files = []
        geometry_files = dict(FUNDAMENTAL="matches.f.txt",ESSENTIAL="matches.e.txt",HOMOGRAPHY="matches.h.txt")
#        for geometric_model in ['FUNDAMENTAL','ESSENTIAL','HOMOGRAPHY']:
        for geometric_model in ['ESSENTIAL']:
            geometry_file = geometry_files.get(geometric_model,"INVALID")
            run_compute_matches( matches_ratio,geometric_model,geometry_file)
            #run_export_matches_visualization(match_type, match_file)
            match_path = f"data/openmvg/matches/{geometry_file}"
            match_files.append( match_path )
            plot_match_histogram(
                f"{match_path}",
                f"data/visuals/match_histogram_{geometry_file.replace('.txt','.png')}"
            )

        selected_match_file = check_and_select_best_matches(
            match_files,
            min_valid_matches=10,
            min_matches_per_pair=10
        )

     
        # Run SfM
        run_sfm(sfm_engine, selected_match_file)

    # Export sparse ply
    export_sparse_ply(sfm_engine)

    # Convert sfm_data.bin → sfm_data.json
    run_convert_sfm_data_format(sfm_engine)

    verify_minimum_poses( min_poses,sfm_engine )

    # Validate sfm_data.json paths
    validate_sfm_data_paths(sfm_engine)

    # Convert OpenMVG → OpenMVS
    run_openmvg_to_openmvs(sfm_engine)

    # === OPENMVS ===

    run_link_images()

    run_densify_pointcloud()

    num_dense_points = export_dense_ply()
    logger.info(f"scene_dense.ply contains {num_dense_points} points")

    MIN_POINTS_FOR_MESH = 1000
    if num_dense_points < MIN_POINTS_FOR_MESH:
        logger.error(f"❌ Aborting pipeline — too few dense points ({num_dense_points}) for ReconstructMesh.")
        sys.exit(1)

    run_reconstruct_mesh()

    #export_mesh_ply()

    if enable_texturing:
        run_texture_mesh()
        #export_textured_mesh_ply()

    logger.info("✅ OpenMVG + OpenMVS pipeline complete.")

# Entry point
if __name__ == "__main__":
    run_pipeline()
