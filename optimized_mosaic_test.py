"""
Benchmark script for OptimizedMosaicGenerator.

Usage:
- Place this file next to optimized_mosaic_generator.py
- Ensure there is a `tiles` directory with tile images
- Ensure there is a `test_images` directory with PNG images
  (for example, those generated in your profiling notebook)

This script:
- Loads the optimized mosaic generator
- Runs it on multiple image sizes and grid sizes
- Prints timing results
"""

import os
import time
from pathlib import Path

from PIL import Image
import pandas as pd

from optimized_mosaic_generator import OptimizedMosaicGenerator


def measure_runtime(generator: OptimizedMosaicGenerator, image_path: str, grid_size: int) -> float:
    """
    Measure runtime for creating a mosaic using the optimized implementation.

    Parameters
    ----------
    generator : OptimizedMosaicGenerator
        Instance of the optimized generator.
    image_path : str
        Path to the test image.
    grid_size : int
        Grid size (grid_size × grid_size).

    Returns
    -------
    float
        Total time taken in seconds.
    """
    img = Image.open(image_path).convert("RGB")
    generator.grid_size = (grid_size, grid_size)

    start = time.time()
    mosaic = generator.create_mosaic(img)
    end = time.time()

    # Optionally ensure mosaic is actually used so it is not optimized away
    _ = mosaic.shape

    return end - start


def main():
    # Directories
    test_dir = Path("test_images")
    tiles_dir = Path("tiles")

    if not tiles_dir.exists():
        print("Tiles directory 'tiles' not found. Run the main app once or add tiles manually.")
        return

    if not test_dir.exists():
        print("Test images directory 'test_images' not found.")
        print("Create it and add PNG test images before running this script.")
        return

    # Initialize generator once
    generator = OptimizedMosaicGenerator(tile_directory=str(tiles_dir), grid_size=(32, 32))

    grid_sizes = [16, 32, 64]
    results = []

    png_files = sorted([f for f in test_dir.iterdir() if f.suffix.lower() == ".png"])

    if not png_files:
        print("No PNG files found in 'test_images'.")
        return

    print("Running optimized performance tests...\n")

    for file in png_files:
        # Expect filenames like name_256.png → resolution = 256
        parts = file.name.split("_")
        if len(parts) >= 2 and parts[-1].endswith(".png"):
            res_str = parts[-1].replace(".png", "")
            try:
                resolution = int(res_str)
            except ValueError:
                resolution = None
        else:
            resolution = None

        for g in grid_sizes:
            print(f"Processing {file.name} at grid {g}×{g}...")
            t = measure_runtime(generator, str(file), g)

            results.append(
                {
                    "image": file.name,
                    "resolution": resolution,
                    "grid_size": g,
                    "runtime_sec": t,
                }
            )

    df = pd.DataFrame(results)
    print("\nOptimized timing results:")
    print(df.to_string(index=False))

    # Save to CSV for notebook / report
    df.to_csv("optimized_timing_results.csv", index=False)
    print("\nSaved optimized timing results to optimized_timing_results.csv")


if __name__ == "__main__":
    main()