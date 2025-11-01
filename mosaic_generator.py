"""
Lab 1 Reference Implementation: Image Mosaic Generator
========================================================

This is a BASIC, INTENTIONALLY INEFFICIENT implementation for students
who need a starting point for Lab 5.

WARNING: This code uses nested loops and is NOT optimized.
Your job in Lab 5 is to profile this code and make it much faster!

Author: Reference Implementation for CS 5130
Date: Fall 2025
"""

import numpy as np
import gradio as gr
from PIL import Image
import os
from pathlib import Path


class SimpleMosaicGenerator:
    """
    A basic mosaic generator that replaces image regions with tiles.

    NOTE: This implementation is intentionally simple and uses nested loops.
    It serves as a baseline for optimization in Lab 5.
    """

    def __init__(self, tile_directory, grid_size=(32, 32)):
        """
        Initialize the mosaic generator.

        Parameters:
        -----------
        tile_directory : str
            Path to directory containing tile images
        grid_size : tuple
            Number of tiles in (height, width) format
        """
        self.tile_directory = tile_directory
        self.grid_size = grid_size
        self.tiles = []
        self.tile_colors = []
        self.load_tiles()

    def load_tiles(self):
        """
        Load all tiles from the tile directory.

        NOTE: This loads tiles every time - inefficient!
        """
        tile_path = Path(self.tile_directory)

        if not tile_path.exists():
            print(f"Warning: Tile directory {self.tile_directory} not found!")
            print("Creating sample tiles...")
            self.create_sample_tiles()
            tile_path = Path(self.tile_directory)

        # Load all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        tile_files = [f for f in tile_path.iterdir()
                      if f.suffix.lower() in valid_extensions]

        if not tile_files:
            print("No tile images found! Creating sample tiles...")
            self.create_sample_tiles()
            tile_files = [f for f in tile_path.iterdir()
                          if f.suffix.lower() in valid_extensions]

        print(f"Loading {len(tile_files)} tiles...")

        for tile_file in tile_files:
            try:
                tile = Image.open(tile_file)
                tile = tile.convert('RGB')
                tile = tile.resize((32, 32))  # Standard tile size
                tile_array = np.array(tile)
                self.tiles.append(tile_array)

                # Compute average color for this tile
                avg_color = np.mean(tile_array, axis=(0, 1))
                self.tile_colors.append(avg_color)
            except Exception as e:
                print(f"Error loading {tile_file}: {e}")

        print(f"Loaded {len(self.tiles)} tiles successfully")

    def create_sample_tiles(self):
        """
        Create sample colored tiles if no tiles are provided.
        This creates a basic set of tiles with different colors.
        """
        os.makedirs(self.tile_directory, exist_ok=True)

        # Create tiles with different colors
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 255),  # White
            (128, 128, 128),  # Gray
            (0, 0, 0),  # Black
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 128),  # Teal
        ]

        for i, color in enumerate(colors):
            tile = np.full((32, 32, 3), color, dtype=np.uint8)
            img = Image.fromarray(tile)
            img.save(os.path.join(self.tile_directory, f'tile_{i:02d}.png'))

        print(f"Created {len(colors)} sample tiles in {self.tile_directory}")

    def preprocess_image(self, image, target_size=512):
        """
        Preprocess the input image: resize and ensure it's RGB.

        Parameters:
        -----------
        image : PIL.Image or np.ndarray
            Input image
        target_size : int
            Target size for the image (will be square)

        Returns:
        --------
        np.ndarray : Preprocessed image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to target size (square)
        image = image.resize((target_size, target_size))

        return np.array(image)

    def divide_into_grid(self, image):
        """
        Divide image into a grid of cells.

        NOTE: This uses nested loops - inefficient!

        Parameters:
        -----------
        image : np.ndarray
            Input image

        Returns:
        --------
        list : List of cell information (row, col, cell_array, avg_color)
        """
        height, width = image.shape[:2]
        grid_h, grid_w = self.grid_size

        cell_height = height // grid_h
        cell_width = width // grid_w

        cells = []

        # INTENTIONALLY INEFFICIENT: Using nested loops
        for i in range(grid_h):
            for j in range(grid_w):
                # Extract cell
                start_h = i * cell_height
                end_h = (i + 1) * cell_height
                start_w = j * cell_width
                end_w = (j + 1) * cell_width

                cell = image[start_h:end_h, start_w:end_w]

                # Compute average color of cell
                avg_color = self.compute_average_color(cell)

                cells.append({
                    'row': i,
                    'col': j,
                    'cell': cell,
                    'avg_color': avg_color
                })

        return cells

    def compute_average_color(self, cell):
        """
        Compute the average color of a cell.

        NOTE: This could be vectorized!

        Parameters:
        -----------
        cell : np.ndarray
            Image cell

        Returns:
        --------
        np.ndarray : Average RGB color
        """
        # INEFFICIENT: Using Python sum instead of NumPy
        total_pixels = cell.shape[0] * cell.shape[1]

        sum_r = 0.0  # Use float to avoid integer overflow
        sum_g = 0.0
        sum_b = 0.0

        # INTENTIONALLY INEFFICIENT: Nested loops
        for i in range(cell.shape[0]):
            for j in range(cell.shape[1]):
                sum_r += float(cell[i, j, 0])
                sum_g += float(cell[i, j, 1])
                sum_b += float(cell[i, j, 2])

        avg_r = sum_r / total_pixels
        avg_g = sum_g / total_pixels
        avg_b = sum_b / total_pixels

        return np.array([avg_r, avg_g, avg_b])

    def find_best_tile(self, target_color):
        """
        Find the tile that best matches the target color.

        NOTE: This uses a loop - could be vectorized!

        Parameters:
        -----------
        target_color : np.ndarray
            Target RGB color

        Returns:
        --------
        int : Index of best matching tile
        """
        best_idx = 0
        best_distance = float('inf')

        # INTENTIONALLY INEFFICIENT: Loop through all tiles
        for i, tile_color in enumerate(self.tile_colors):
            # Compute Euclidean distance
            distance = 0
            for c in range(3):  # R, G, B
                distance += (target_color[c] - tile_color[c]) ** 2
            distance = distance ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_idx = i

        return best_idx

    def create_mosaic(self, image):
        """
        Create a mosaic from the input image.

        Parameters:
        -----------
        image : PIL.Image or np.ndarray
            Input image

        Returns:
        --------
        np.ndarray : Mosaic image
        """
        # Preprocess image
        processed = self.preprocess_image(image)
        print(f"Preprocessed image shape: {processed.shape}")
        print(f"Image value range: {processed.min()} to {processed.max()}")

        # Divide into grid
        cells = self.divide_into_grid(processed)
        print(f"Created {len(cells)} cells")

        # Create output image
        height, width = processed.shape[:2]
        mosaic = np.zeros_like(processed)

        grid_h, grid_w = self.grid_size
        cell_height = height // grid_h
        cell_width = width // grid_w

        print(f"Creating mosaic with {len(cells)} cells...")
        print(f"Cell dimensions: {cell_height}×{cell_width}")

        # Replace each cell with best matching tile
        # INTENTIONALLY INEFFICIENT: Loop through cells
        for idx, cell_info in enumerate(cells):
            if idx == 0:  # Debug first cell
                print(f"First cell avg color: {cell_info['avg_color']}")

            i = cell_info['row']
            j = cell_info['col']
            avg_color = cell_info['avg_color']

            # Find best matching tile
            tile_idx = self.find_best_tile(avg_color)

            if idx == 0:  # Debug first tile match
                print(f"Matched to tile {tile_idx} with color {self.tile_colors[tile_idx]}")

            tile = self.tiles[tile_idx]

            # Resize tile to match cell size
            tile_resized = Image.fromarray(tile).resize((cell_width, cell_height))
            tile_resized = np.array(tile_resized)

            # Place tile in mosaic
            start_h = i * cell_height
            end_h = (i + 1) * cell_height
            start_w = j * cell_width
            end_w = (j + 1) * cell_width

            mosaic[start_h:end_h, start_w:end_w] = tile_resized

        print(f"Mosaic complete. Value range: {mosaic.min()} to {mosaic.max()}")
        return mosaic

    def compute_mse(self, original, mosaic):
        """
        Compute Mean Squared Error between original and mosaic.

        Parameters:
        -----------
        original : np.ndarray
            Original image
        mosaic : np.ndarray
            Mosaic image

        Returns:
        --------
        float : MSE value
        """
        original = self.preprocess_image(original)
        mse = np.mean((original.astype(float) - mosaic.astype(float)) ** 2)
        return mse


def create_gradio_interface():
    """
    Create a Gradio interface for the mosaic generator.
    """

    # Initialize generator (will create sample tiles if needed)
    generator = SimpleMosaicGenerator(
        tile_directory='tiles',
        grid_size=(32, 32)
    )

    def process_image(input_image, grid_size):
        """
        Process image and return mosaic.

        Parameters:
        -----------
        input_image : PIL.Image
            Input image from Gradio
        grid_size : int
            Grid size (will be grid_size × grid_size)

        Returns:
        --------
        tuple : (mosaic_image, mse_value, message)
        """
        try:
            # Update grid size
            generator.grid_size = (grid_size, grid_size)

            # Create mosaic
            mosaic = generator.create_mosaic(input_image)

            # Compute MSE
            mse = generator.compute_mse(input_image, mosaic)

            message = f"Mosaic created successfully!\nGrid: {grid_size}×{grid_size}\nMSE: {mse:.2f}"

            return mosaic, message

        except Exception as e:
            return None, f"Error: {str(e)}"

    # Create Gradio interface
    with gr.Blocks(title="Image Mosaic Generator") as demo:
        gr.Markdown("# 🎨 Image Mosaic Generator")
        gr.Markdown("Upload an image and create a mosaic version using colored tiles!")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                grid_size = gr.Slider(
                    minimum=8,
                    maximum=64,
                    value=32,
                    step=8,
                    label="Grid Size (cells per side)"
                )
                generate_btn = gr.Button("Generate Mosaic", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Mosaic Output")
                output_text = gr.Textbox(label="Status", lines=3)

        # Connect components
        generate_btn.click(
            fn=process_image,
            inputs=[input_image, grid_size],
            outputs=[output_image, output_text]
        )

        # Add examples if available
        gr.Markdown("### 💡 Tips")
        gr.Markdown("""
        - Start with smaller grid sizes (16×16) for faster processing
        - Larger grid sizes create more detailed mosaics but take longer
        - The default tiles are simple colored squares - you must add your own tiles to the 'tiles' folder!
        """)

    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch()