"""
Lab 5 – Optimized Implementation: Image Mosaic Generator

Optimized changes:
- Vectorized grid operations (no nested loops)
- Vectorized average color computation
- Vectorized tile matching via NumPy broadcasting
- Tiles loaded once and cached in memory
- Tiles pre-resized once per grid size
- Silent execution, no debug prints
"""

import numpy as np
from PIL import Image
from pathlib import Path
import gradio as gr


class OptimizedMosaicGenerator:
    """
    High performance mosaic generator compatible with the original API.
    """

    def __init__(self, tile_directory: str, grid_size=(32, 32)):
        self.tile_directory = tile_directory
        self.grid_size = grid_size

        self.tiles = None
        self.tile_colors = None

        self.resized_tiles_cache = {}

        self.load_tiles()

    def load_tiles(self):
        """
        Load tiles once and precompute their average colors.
        """
        tile_path = Path(self.tile_directory)

        if not tile_path.exists():
            raise RuntimeError(f"Tile directory {self.tile_directory} does not exist.")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        tile_files = [f for f in tile_path.iterdir() if f.suffix.lower() in valid_extensions]

        if not tile_files:
            raise RuntimeError("No tile images found in tile directory.")

        tiles_list = []
        for tile_file in tile_files:
            tile = Image.open(tile_file).convert("RGB")
            tile = tile.resize((32, 32), Image.NEAREST)
            tiles_list.append(np.array(tile, dtype=np.uint8))

        self.tiles = np.stack(tiles_list, axis=0)
        self.tile_colors = self.tiles.mean(axis=(1, 2))

        self.resized_tiles_cache.clear()

    def preprocess_image(self, image, target_size=512):
        """
        Convert image to RGB and resize to 512×512.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((target_size, target_size), Image.NEAREST)
        return np.array(image, dtype=np.uint8)

    def _compute_grid_shapes(self, image_array):
        height, width = image_array.shape[:2]
        grid_h, grid_w = self.grid_size

        cell_h = height // grid_h
        cell_w = width // grid_w

        return grid_h, grid_w, cell_h, cell_w

    def _extract_cells_and_colors(self, image_array):
        grid_h, grid_w, cell_h, cell_w = self._compute_grid_shapes(image_array)

        cells = image_array.reshape(
            grid_h, cell_h, grid_w, cell_w, 3
        ).transpose(0, 2, 1, 3, 4)

        cell_colors = cells.mean(axis=(2, 3))

        return cells, cell_colors

    def _get_resized_tiles_for_cell_size(self, cell_h, cell_w):
        """
        Cache resized tiles so they are computed only once per grid size.
        """
        key = (cell_h, cell_w)
        if key in self.resized_tiles_cache:
            return self.resized_tiles_cache[key]

        num_tiles = self.tiles.shape[0]
        resized = np.empty((num_tiles, cell_h, cell_w, 3), dtype=np.uint8)

        for idx in range(num_tiles):
            tile_img = Image.fromarray(self.tiles[idx])
            tile_resized = tile_img.resize((cell_w, cell_h), Image.NEAREST)
            resized[idx] = np.array(tile_resized, dtype=np.uint8)

        self.resized_tiles_cache[key] = resized
        return resized

    def _match_cells_to_tiles(self, cell_colors_flat):
        diffs = cell_colors_flat[:, None, :] - self.tile_colors[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        return np.argmin(dists, axis=1)

    def compute_average_color(self, cell):
        return cell.mean(axis=(0, 1))

    def find_best_tile(self, target_color):
        diffs = self.tile_colors - target_color[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    def create_mosaic(self, image):
        processed = self.preprocess_image(image)

        grid_h, grid_w, cell_h, cell_w = self._compute_grid_shapes(processed)
        _, cell_colors = self._extract_cells_and_colors(processed)

        num_cells = grid_h * grid_w
        cell_colors_flat = cell_colors.reshape(num_cells, 3)

        best_tile_indices = self._match_cells_to_tiles(cell_colors_flat)
        resized_tiles = self._get_resized_tiles_for_cell_size(cell_h, cell_w)

        mosaic = np.zeros_like(processed, dtype=np.uint8)

        mosaic_cells = mosaic.reshape(
            grid_h, cell_h, grid_w, cell_w, 3
        ).transpose(0, 2, 1, 3, 4)

        tile_indices_grid = best_tile_indices.reshape(grid_h, grid_w)

        tiles_per_cell = resized_tiles[tile_indices_grid.reshape(-1)]
        tiles_per_cell = tiles_per_cell.reshape(grid_h, grid_w, cell_h, cell_w, 3)

        mosaic_cells[...] = tiles_per_cell

        return mosaic_cells.transpose(0, 2, 1, 3, 4).reshape(processed.shape)

    def compute_mse(self, original, mosaic):
        original_proc = self.preprocess_image(original)
        return float(np.mean((original_proc.astype(float) - mosaic.astype(float)) ** 2))


def create_gradio_interface():
    generator = OptimizedMosaicGenerator(
        tile_directory="tiles",
        grid_size=(32, 32),
    )

    # Load example images from sample_images/
    example_dir = Path("sample_images")
    example_paths = []
    if example_dir.exists():
        for f in sorted(example_dir.iterdir()):
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                example_paths.append(str(f))

    def process_image(input_image, grid_size):
        try:
            generator.grid_size = (grid_size, grid_size)
            mosaic = generator.create_mosaic(input_image)
            mse = generator.compute_mse(input_image, mosaic)

            message = f"Mosaic created.\nGrid: {grid_size}×{grid_size}\nMSE: {mse:.2f}"
            return mosaic, message

        except Exception as e:
            return None, f"Error: {str(e)}"

    with gr.Blocks(title="Optimized Image Mosaic Generator") as demo:
        gr.Markdown("# 🎨 Optimized Image Mosaic Generator")
        gr.Markdown("Faster version using NumPy vectorization.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")

                if example_paths:
                    gr.Markdown("### Example Images")
                    gr.Examples(
                        examples=example_paths,
                        inputs=input_image,
                        label="Click an image below to load it"
                    )

                grid_size = gr.Slider(8, 64, value=32, step=8, label="Grid Size")
                generate_btn = gr.Button("Generate Mosaic")

            with gr.Column():
                output_image = gr.Image(label="Mosaic Output")
                output_text = gr.Textbox(label="Status", lines=3)

        generate_btn.click(
            fn=process_image,
            inputs=[input_image, grid_size],
            outputs=[output_image, output_text],
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()