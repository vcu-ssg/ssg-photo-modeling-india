
import os
import subprocess
from pathlib import Path
import glob
import concurrent.futures
import time


DJI_FOCAL_LENGTHS = {
    "DJI FC3682": {
        "FocalLength": 4.5,  # in mm
        "FocalLengthIn35mmFormat": 24
    },
    # You can add more models here in future.
}

def get_camera_model_from_mp4(video_path):
    """Extract 'Model' from MP4 using exiftool"""
    try:
        result = subprocess.run(
            ["exiftool", "-Model", "-s3", video_path],
            capture_output=True,
            text=True,
            check=True
        )
        model = result.stdout.strip()
        if not model:
            print(f"Warning: No Model tag found in {video_path}")
            return "Unknown"
        print(f"Camera Model: {model}")
        return model
    except subprocess.CalledProcessError as e:
        print(f"Error extracting Model from {video_path}: {e}")
        return "Unknown"

def add_camera_model_to_images(file_list, camera_model):
    """Write EXIF tags (Model, CameraModelName, FocalLength) to specific files only."""
    if not file_list:
        print("No images to tag.")
        return

    # Normalize camera_model to "DJI FCxxxx" format if needed
    if not camera_model.startswith("DJI "):
        camera_model = "DJI " + camera_model

    # Lookup focal length data
    focal_data = DJI_FOCAL_LENGTHS.get(camera_model)
    if not focal_data:
        print(f"Warning: No known focal length for model {camera_model}. Will skip FocalLength tags.")
        focal_tags = []
    else:
        focal_tags = [
            f"-FocalLength={focal_data['FocalLength']}",
            f"-FocalLengthIn35mmFormat={focal_data['FocalLengthIn35mmFormat']}"
        ]

    if ' ' in camera_model:
        make, model = camera_model.split(' ', 1)
    else:
        # Fallback if no space (just in case)
        make = "Unknown"
        model = camera_model
    
    # Build exiftool command
    cmd = [
        "exiftool",
        f"-Model={model}",
        f"-Make={make}",
        *focal_tags,
        "-overwrite_original",
        *file_list
    ]
    print(f"Tagging {len(file_list)} images with Model={camera_model}")
    subprocess.run(cmd, check=True)



def extract_frames_from_file(video_path, output_dir, fps=1.0, skip_seconds = 5, threads=8, quality=2, capture_seconds=None, 
                             format="jpg",tag="tag",max_width=1600):
    os.makedirs(output_dir, exist_ok=True)
    """ extract frames from file """
    
    video_name = Path(video_path).stem
    output_template = os.path.join(output_dir, f"{tag}_%05d.{format}")
    file_mask = f"{tag}_*.{format}"
    
    # Step 1: Record existing files before extraction
    existing_files = set(Path(output_dir).glob(file_mask))
    existing_files = ()

    # Step 2: Run ffmpeg to extract frames
    cmd = [
        "ffmpeg",
        "-threads", str(threads),
#        "-noautorotate",
        ]
    if capture_seconds:
        cmd.append("-t")
        cmd.append(str(capture_seconds))
       
    cmd.extend( [
        "-ss", str(skip_seconds),
        "-i", video_path ] )
               
    if format=="jpg":
        cmd.extend([f"-q:v", str(quality)] )
    else:
        pass

    cmd.extend(["-vf", f"fps={fps},format=yuv420p,scale='if(gt(iw,ih),min({max_width},iw),-2)':'if(gt(ih,iw),min({max_width},ih),-2)'" ])
                      
    cmd.append( output_template )
    
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Step 3: Identify newly created files
    all_files = set(Path(output_dir).glob(file_mask))
    #new_files = sorted(list(all_files - existing_files))
    new_files = all_files

    # Step 4: Get camera model and tag only the new files
    camera_model = get_camera_model_from_mp4(video_path)
    add_camera_model_to_images([str(p) for p in new_files], camera_model)

def extract_frames_from_folder(video_dir, output_dir, fps):
    video_dir = Path(video_dir)
    for video_file in video_dir.glob("*.MP4"):
        print(f"\n=== Extracting frames from {video_file} ===")
        extract_frames_from_file(str(video_file), output_dir, fps)
        

def process_image_with_convert(
    file_name,
    output_folder,
    sharpen="0x1.0",
    contrast="5x50%",
    greyscale=False,
    crop=None,
    tag="filtered"
):
    """Run ImageMagick convert on a single file, outputting to a folder, with configurable sharpen, contrast, greyscale, and crop."""
    
    if not file_name or not os.path.isfile(file_name):
        print(f"File not found: {file_name}")
        return

    os.makedirs(output_folder, exist_ok=True)

    img_name = os.path.basename(file_name)
    output_path = os.path.join(output_folder, img_name)

    # DJI_0150-filtered_jpg_0.20_800_00001.png
    
    part1, part2 = img_name.rsplit("_",1)

    img_name = "_".join([tag,part2])
    output_path = os.path.join(output_folder, img_name)

    cmd = [
        "convert",
        file_name,
        "-auto-level",
        "-sigmoidal-contrast", contrast,
        "-sharpen", sharpen
    ]

    # Optional crop
    if crop:
        cmd.extend(["-crop", crop])

    # Optional greyscale
    if greyscale:
        cmd.extend(["-colorspace", "Gray"])

    # Output
    cmd.append(output_path)

    print(f"Processing: {img_name} → {output_path}")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Done.")


def process_folder_with_convert(
    input_folder,
    output_folder,
    sharpen="0x1.0",
    contrast="5x50%",
    greyscale=False,
    crop=None,
    tag="filtered"
):
    """Process all .jpg files in input_folder using process_image_with_convert."""

    parts = tag.split("-",1)
    parts = parts[1].split("_",4)
    # Glob all jpg files in input_folder
    file_list = sorted(glob.glob(os.path.join(input_folder, f"*.{parts[1]}")))

    if not file_list:
        print(f"No {parts[1]} images found in {input_folder}")
        return

    print(f"Processing {len(file_list)} images from {input_folder} → {output_folder}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file
    for file_name in file_list:
        process_image_with_convert(
            file_name,
            output_folder,
            sharpen=sharpen,
            contrast=contrast,
            greyscale=greyscale,
            crop=crop,
            tag=tag
        )

    print("All images processed.")


def process_folder_with_convert_workers(
    input_folder,
    output_folder,
    sharpen="0x1.0",
    contrast="5x50%",
    greyscale=False,
    crop=None,
    tag="filtered",
    max_workers=8,
    format="png"
):
    """Process all files in input_folder using process_image_with_convert (parallel), with timing."""

    start_time = time.time()

    # Glob all jpg files in input_folder
    file_list = sorted(glob.glob(os.path.join(input_folder, f"*.{format}")))
    
    if not file_list:
        print(f"No {parts[1]} images found in {input_folder}")
        return

    print(f"Processing {len(file_list)} images from {input_folder} → {output_folder} (parallel with {max_workers} workers)")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define wrapper function for executor
    def process_single_file(file_name):
        process_image_with_convert(
            file_name,
            output_folder,
            sharpen=sharpen,
            contrast=contrast,
            greyscale=greyscale,
            crop=crop,
            tag=tag
        )

    # Use ThreadPoolExecutor to process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_single_file, file_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("All images processed.")
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
