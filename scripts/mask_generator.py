"""
    
"""

import cv2
import numpy as np
from pathlib import Path

import cv2
import numpy as np
import click

from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_combined_mask(input_image_path, output_mask_path, output_masked_image_path=None):
    img_bgr = cv2.imread(str(input_image_path))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # Upward vertical scan to mask ground/horizon
    h, w = edges.shape
    vertical_mask = np.zeros_like(edges_dilated, dtype=np.uint8)
    min_edge_run = 5

    for x in range(w):
        count = 0
        for y in range(h-1, -1, -1):
            if edges_dilated[y, x] > 0:
                count += 1
                if count >= min_edge_run:
                    vertical_mask[:y+1, x] = 255
                    break
            else:
                count = 0

    # Combine masks
    final_mask = cv2.bitwise_and(edges_dilated, vertical_mask)

    # Write mask
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask_path), final_mask)

    # Apply mask to original image if output_masked_image_path provided
    if output_masked_image_path is not None:
        # Ensure mask is 3-channel for color image masking
        mask_3ch = cv2.merge([final_mask, final_mask, final_mask])
        masked_img = cv2.bitwise_and(img, mask_3ch)
        
        output_masked_image_path.parent.mkdir(parents=True, exist_ok=True)
        masked_img_bgr = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_masked_image_path), masked_img_bgr)

def generate_masks_in_directory(images_dir, output_mask_dir, output_masked_image_dir=None, workers=8, filter="default"):
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    if output_masked_image_dir:
        output_masked_image_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        click.echo("No PNG images found.")
        return

    click.echo(f"Generating masks for {len(png_files)} images using {workers} threads and filter '{filter}'...")

    def process_image(img_path):
        mask_path = output_mask_dir / (img_path.stem + ".png")
        masked_image_path = (output_masked_image_dir / (img_path.name)) if output_masked_image_dir else None

        if filter.lower() == "none":
            # create a fully white mask (keep everything)
            img_bgr = cv2.imread(str(img_path))
            h, w = img_bgr.shape[:2]
            full_mask = np.full((h, w), 255, dtype=np.uint8)
            cv2.imwrite(str(mask_path), full_mask)

            if masked_image_path:
                cv2.imwrite(str(masked_image_path), img_bgr)

        elif filter.lower() == "default":
            # your existing edge + vertical mask strategy
            generate_combined_mask(img_path, mask_path, masked_image_path)

        else:
            raise ValueError(f"Unknown filter type '{filter}'")

        return mask_path.name

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_image, img_path): img_path for img_path in png_files}

        for future in as_completed(futures):
            img_name = futures[future].name
            try:
                mask_name = future.result()
                click.echo(f"✅ {mask_name} created.")
            except Exception as e:
                click.echo(f"❌ Error processing {img_name}: {e}")

    click.echo(f"🎉 All masks written to: {output_mask_dir}")
    if output_masked_image_dir:
        click.echo(f"🖼 Masked images written to: {output_masked_image_dir}")
