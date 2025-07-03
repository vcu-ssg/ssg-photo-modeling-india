import click
from scripts.extract_frames import extract_frames_from_file, extract_frames_from_folder, process_folder_with_convert, process_folder_with_convert_workers
from scripts import colmap_pipeline, openmvg_pipeline, convert_matches_g_to_dot, gsplat_pipeline, openmvs_pipeline
from scripts.mask_generator import generate_combined_mask,generate_masks_in_directory

from scripts.report_utils import build_folder_tree_with_files, write_qmds_from_tree

import torch
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

@click.group()
def cli():
    """GBT 3D Pipeline CLI"""
    pass

@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output-dir", default="data/frames", help="Output dir for extracted frames")
@click.option("--fps", type=float, default=1.0, help="Frames per second to extract")
@click.option("--skip", type=int, default=5, help="Skip first X seconds")
@click.option("--capture", type=int, default=None, help="Capture X seconds of video")
@click.option("--threads", type=int, default=8, help="FFMPEG threads to use")
@click.option("--quality", type=int, default=2, help="JPG quality 1/31")
@click.option("--tag", default="tag", help="filename tag")
@click.option("--format", default="png", help="Output file format (png/jpg)")
@click.option("--max_width", default="1600", help="max image width")
def extract_frames(video_path, output_dir,fps,skip,capture,threads,quality,tag,format,max_width):
    """Extract frames from video"""
    extract_frames_from_file(video_path, output_dir, fps=fps,skip_seconds=str(skip),
        threads=threads,quality=quality,capture_seconds=capture,tag=tag,format=format, max_width=max_width)


@cli.command()
@click.option("--frame-dir", default="data/frames", help="Dir with frames to process")
@click.option("--colmap-dir", default="data/colmap", help="COLMAP output dir")
@click.option("--output-format", default="ply", type=click.Choice(["ply", "obj"]), help="Output format")
def run_colmap(frame_dir, colmap_dir, output_format):
    """Run COLMAP pipeline"""
    colmap_pipeline.run_pipeline(frame_dir, colmap_dir, output_format)

@cli.command()
@click.option('--enable-texturing/--no-enable-texturing', is_flag=True, default=False, help='Enable/disable OpenMVS TextureMesh step.')
@click.option('--sfm-engine', type=click.Choice(['GLOBAL', 'INCREMENTAL'], case_sensitive=False), default='GLOBAL', help='SfM engine to use.')
@click.option('--matches-ratio', type=float, default=0.6, show_default=True, help='Feature matching ratio filter (lower = more matches).')
def run_openmvg_openmvs(enable_texturing, sfm_engine, matches_ratio):
    """Run OpenMVG + OpenMVS pipeline"""
    print("==== Pipeline configuration ====")
    print(f" SfM engine      : {sfm_engine}")
    print(f" Matches ratio   : {matches_ratio}")
    print(f" Texturing       : {'ENABLED' if enable_texturing else 'DISABLED'}")
    print("================================\n")    

    # Call run_pipeline with correct args:
    openmvg_pipeline.run_pipeline(
        enable_texturing=enable_texturing,
        sfm_engine=sfm_engine.upper(),
        matches_ratio=matches_ratio
    )
    
@cli.command()
@click.option(
    "--input", "-i", required=True, type=click.Path(exists=True, dir_okay=False),
    help="Path to matches.g.txt file"
)
@click.option(
    "--output", "-o", required=True, type=click.Path(dir_okay=False),
    help="Path to output DOT file"
)
def convert_matches_g(input, output):
    """Convert matches.g.txt (edge list) → DOT format for visualization."""
    click.echo(f"👉 Converting {input} → {output} ...")
    convert_matches_g_to_dot.convert_matches_g_to_dot(input, output)
    click.echo(f"✅ Done. DOT file written to: {output}")


@cli.command()
@click.argument('matches_file', type=click.Path(exists=True))
@click.option('--show-disconnected', is_flag=True, help='Print list of disconnected nodes.')
def analyze_graph(matches_file, show_disconnected):
    """ Analyze dot file """
    convert_matches_g_to_dot.analyze_graph(matches_file, show_disconnected)


@cli.command()
@click.option("--input-folder", required=True, type=click.Path(exists=True, file_okay=False), help="Input folder with JPG images")
@click.option("--output-folder", required=True, type=click.Path(file_okay=False), help="Output folder for processed images")
@click.option("--sharpen", default="0x1.0", show_default=True, help="Sharpen amount, e.g. 0x1.0")
@click.option("--contrast", default="5x50%", show_default=True, help="Sigmoidal contrast amount, e.g. 5x50%%")
@click.option("--greyscale/--no-greyscale", default=False, help="Convert to greyscale")
@click.option("--crop", default=None, help="Crop geometry, e.g. WxH+X+Y (optional)")
@click.option("--tag", default="filtered", help="tag for file name")
@click.option("--workers", default=8, help="Number of separate workers")
@click.option("--format", default="png", help="Input image format (png/jpg)")
def convert_images(input_folder, output_folder, sharpen, contrast, greyscale, crop, tag,workers,format ):
    """Run ImageMagick convert on all images in input folder."""
    click.echo(f"👉 Converting images in {input_folder} → {output_folder}")
    click.echo(f"  Format    : {format}")
    click.echo(f"  Sharpen   : {sharpen}")
    click.echo(f"  Contrast  : {contrast}")
    click.echo(f"  Greyscale : {'ON' if greyscale else 'OFF'}")
    click.echo(f"  Crop      : {crop if crop else 'None'}")
    click.echo(f"  Tag       : {tag}")

    process_folder_with_convert_workers(
        input_folder,
        output_folder,
        sharpen=sharpen,
        contrast=contrast,
        greyscale=greyscale,
        crop=crop,
        tag=tag,
        max_workers=workers,
        format=format
    )

    click.echo(f"✅ Done. Processed images in: {output_folder}")
    


@cli.command()
@click.option("--image-path", type=click.Path(exists=True, file_okay=False),help="Image folder")
@click.option("--output-model-path", type=click.Path(),help="colmap output folder")
def run_colmap_pipeline_cli(image_path, output_model_path):
    """Run COLMAP pipeline on given image folder."""
    from scripts.colmap_pipeline import run_colmap_pipeline
    run_colmap_pipeline(image_path, output_model_path)


@cli.command()
@click.option("--input-model-folder", required=True, type=click.Path(exists=True, file_okay=False),
              help="Path to the input COLMAP model folder (e.g., sparse/0)")
@click.option("--output-model-folder", required=True, type=click.Path(),
              help="Path to the output cleaned model folder (e.g., sparse/clean)")
@click.option("--min-track-len", default=2, show_default=True, type=int,
              help="Minimum number of observations per 3D point")
@click.option("--max-reproj-error", default=4.0, show_default=True, type=float,
              help="Maximum allowed reprojection error")
@click.option("--min-tri-angle", default=1.5, show_default=True, type=float,
              help="Maximum allowed reprojection error")
def colmap_model_cleaner(input_model_folder, output_model_folder, min_track_len, max_reproj_error, min_tri_angle):
    """Clean a COLMAP sparse model using model_cleaner."""
    colmap_pipeline.run_colmap_point_filtering(
        input_model_host=input_model_folder,
        output_model_host=output_model_folder,
        min_track_len=min_track_len,
        max_reproj_error=max_reproj_error,
        min_tri_angle=min_tri_angle
    )


@cli.command()
@click.option('--scene', required=True, help='Scene name, e.g. DJI_0145-FPS-1.60-original')
@click.option('--images-dir', required=True, type=click.Path(), help='Path to input images')
@click.option('--sparse-dir', required=True, type=click.Path(), help='Path to sparse COLMAP model (sparse/0)')
@click.option('--model-dir', required=True, type=click.Path(), help='Output directory for gsplat results')
@click.option('--iterations', required=True, help='Iterations to gsplat')
@click.option('--sh_degree', required=True, help='Spherical Harmonics degree. 0-none, 1-2 lower res, 3-4 higher res')
def run_gsplat_pipeline(scene, images_dir, sparse_dir, model_dir,iterations,sh_degree):
    """Run Gaussian Splatting training for a specific scene with provided paths."""
    gsplat_pipeline.run_gsplat_pipeline(scene, images_dir, sparse_dir, model_dir,iterations,sh_degree)

def ensure_absolute_path(input_model_folder):
    if not input_model_folder.startswith("/"):
        input_model_folder = "/" + input_model_folder
    return input_model_folder

@cli.command()
@click.option("--input-model-folder", type=click.Path(exists=True, file_okay=False),help="Input model folder")
@click.option("--output-stats-folder", type=click.Path(),help="Output stats folder")
def run_colmap_model_analyzer(input_model_folder, output_stats_folder ):
    """Run COLMAP analyzer on folders.  Not necessary unless format of output has changed.  Pipeline calls it, too."""
    from scripts.colmap_pipeline import run_colmap_model_analyzer
    run_colmap_model_analyzer( ensure_absolute_path(input_model_folder), output_stats_folder, 0.0 )

@cli.command()
@click.option("--input-file", type=click.Path(exists=True, file_okay=True),help="Input .ply file")
@click.option("--output-file", type=click.Path(),help="Output .splat file")
def run_ply_to_splat_converter( input_file, output_file ):
    """splat converter"""
    from gsplat_pipeline import run_ply_to_splat_converter
    run_ply_to_splat_converter( ensure_absolute_path(input_file), ensure_absolute_path(output_file) )



@cli.command()
@click.option('--input-file', type=click.Path(exists=True), help="input file")
@click.option('--output-file', type=click.Path(), help="output file")
@click.option('--zmin', type=float, default=None, help='Minimum Z coordinate to keep')
@click.option('--zmax', type=float, default=None, help='Maximum Z coordinate to keep')
@click.option('--min-opacity', type=float, default=0.0, help='Minimum opacity threshold')
@click.option('--max-scale-x', type=float, default=None, help='Maximum X scale threshold')
@click.option('--max-scale-y', type=float, default=None, help='Maximum Y scale threshold')
@click.option('--max-scale-z', type=float, default=None, help='Maximum Z scale threshold')
def run_splat_post_cleaner(input_file, output_file, zmin, zmax, min_opacity, max_scale_x, max_scale_y, max_scale_z):
    """Filter a .splat file to remove foggy/outlier Gaussians by Z range, opacity, and scale."""

    click.echo(f"Loading splat file: {input_file}")
    state = torch.load(input_file,weights_only=False)

    keep = torch.ones(state['means3D'].shape[0], dtype=torch.bool)

    if zmin is not None or zmax is not None:
        z = state['means3D'][:, 2]
        if zmin is not None:
            keep &= (z > zmin)
        if zmax is not None:
            keep &= (z < zmax)

    if min_opacity > 0.0:
        opacity = state['opacity']
        keep &= (opacity >= min_opacity)

    if any(val is not None for val in [max_scale_x, max_scale_y, max_scale_z]):
        scales = state['scales']
        if max_scale_x is not None:
            keep &= (scales[:, 0] <= max_scale_x)
        if max_scale_y is not None:
            keep &= (scales[:, 1] <= max_scale_y)
        if max_scale_z is not None:
            keep &= (scales[:, 2] <= max_scale_z)

    original_count = keep.numel()
    kept_count = keep.sum().item()
    click.echo(f"Filtering complete: Kept {kept_count} / {original_count} Gaussians")

    for key in ['means3D', 'opacity', 'scales', 'colors_precomp', 'rotation']:
        if key in state:
            state[key] = state[key][keep]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_file)
    click.echo(f"Saved filtered splat to: {output_file}")


@cli.command()
@click.option("--image-folder", required=True, type=click.Path(exists=True, file_okay=False),
              help="Path to undistorted images folder (usually output of COLMAP undistorter)")
@click.option("--sparse-model-folder", required=True, type=click.Path(exists=True, file_okay=False),
              help="Path to COLMAP sparse model folder (e.g., sparse/0 or sparse/0_filter1)")
@click.option("--mvs-output-folder", required=True, type=click.Path(file_okay=False),
              help="Output directory for OpenMVS intermediate and final files")
def run_openmvs_pipeline(image_folder, sparse_model_folder, mvs_output_folder):
    """Run full OpenMVS pipeline and export web-ready GLB mesh."""
    from scripts.openmvs_pipeline import mvs_pipeline
    mvs_pipeline(image_folder, sparse_model_folder, mvs_output_folder)


@cli.command()
@click.option("--input-file", required=True, type=click.Path(exists=True), help="Input mesh file (.ply)")
@click.option("--output-file", required=True, type=click.Path(), help="Output cleaned mesh file")
@click.option("--min-component-diag", default=0.2, type=float, help="Minimum component diagonal to keep")
@click.option("--recompute-normals/--no-recompute-normals", default=True, help="Recompute normals for the mesh")
@click.option("--k-neighbors", default=12, type=int, help="Number of neighbors for normal estimation")
@click.option("--remove-duplicates/--no-remove-duplicates", default=True, help="Remove duplicate vertices")
@click.option("--remove-unref/--no-remove-unref", default=True, help="Remove unreferenced vertices")
@click.option("--remove-zero-area/--no-remove-zero-area", default=True, help="Remove zero area faces")
@click.option("--binary/--ascii", default=True, help="Save mesh in binary format (default) or ASCII")
def clean_dense_mesh_cmd(input_file, output_file, min_component_diag, recompute_normals, k_neighbors,
                         remove_duplicates, remove_unref, remove_zero_area, binary):
    """Clean a dense mesh with PyMeshLab"""
    from scripts.pymeshlab_utils import clean_dense_mesh
    clean_dense_mesh(
        input_mesh=Path(input_file),
        output_mesh=Path(output_file),
        min_component_diag=min_component_diag,
        recompute_normals=recompute_normals,
        k_neighbors=k_neighbors,
        remove_duplicates=remove_duplicates,
        remove_unref=remove_unref,
        remove_zero_area=remove_zero_area,
        binary=binary
    )


@cli.command()
@click.option("--images-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Input folder containing PNG frames.")
@click.option("--output-mask-dir", required=True, type=click.Path(file_okay=False),
              help="Output folder for generated mask images.")
def generate_masksxxxx(images_dir, output_mask_dir):
    """Generate combined edge + vertical masks for COLMAP."""
    from scripts.mask_generator import generate_combined_mask

    images_dir = Path(images_dir)
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        click.echo("No PNG images found.")
        return

    click.echo(f"Generating masks for {len(png_files)} images...")

    for img_path in png_files:
        mask_path = output_mask_dir / (img_path.stem + ".png")
        generate_combined_mask(img_path, mask_path)
        click.echo(f"✅ {mask_path.name} created.")

    click.echo(f"🎉 All masks written to: {output_mask_dir}")


@cli.command()
@click.option("--projects-root", required=True, type=click.Path(exists=True, file_okay=False),
              help="Root folder containing all projects.")
@click.option("--report-qmds", required=True, type=click.Path(file_okay=False),
              help="Output folder for generated QMD files.")
@click.option("--report-data", required=True, type=click.Path(file_okay=False),
              help="Output folder for data.")
def generate_project_reports(projects_root, report_qmds, report_data ):
    """Scan projects folder and generate QMD reports."""
    click.echo(f"🔍 Scanning projects in: {projects_root}")
    tree = build_folder_tree_with_files(projects_root)
    click.echo(f"📂 Found {len(tree)} projects.")

    write_qmds_from_tree(tree, report_qmds, report_data )
    click.echo(f"🎉 All QMD reports written to: {report_qmds}")


@cli.command()
@click.option("--images-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Input folder containing PNG frames.")
@click.option("--output-mask-dir", required=True, type=click.Path(file_okay=False),
              help="Output folder for generated mask images.")
@click.option("--output-masked-image-dir", required=False, type=click.Path(file_okay=False),
              help="Optional output folder for masked images (images with mask applied).")
@click.option("--filter", default=None, show_default=True,
              help="Number of threads/workers to use.")
@click.option("--workers", default=8, show_default=True,
              help="Number of threads/workers to use.")
def generate_masks(images_dir, output_mask_dir, output_masked_image_dir, filter, workers):
    """Generate combined edge + vertical masks for COLMAP and optionally masked images."""
    generate_masks_in_directory(
        Path(images_dir),
        Path(output_mask_dir),
        Path(output_masked_image_dir) if output_masked_image_dir else None,
        workers,
        filter
    )

if __name__ == "__main__":
    cli()

