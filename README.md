# Fast Isometric View Generator for 3D Models

A high-performance Python script to generate isometric preview images from 3D models (STL, 3MF, etc.). It leverages parallel processing to rapidly handle large batches of files and offers features for mesh decimation and image optimization.

This tool is ideal for creating consistent, clean, and lightweight thumbnails for 3D model libraries.

![Sample Output Placeholder](https://i.imgur.com/7b7Jg8p.png)
*(This is a sample image showing the style of output)*

---

## Features

-   **Parallel Processing**: Uses multiple CPU cores to process entire directories of models at maximum speed.
-   **Multi-part Geometry Handling**: Automatically arranges multi-body files in a grid to ensure all parts are visible.
-   **Mesh Decimation**: Optionally reduces mesh complexity for faster rendering of very high-poly models.
-   **Image Optimization**: Compresses output images to save space, with support for PNG, JPG, and WebP formats.
-   **Customizable Output**: Control image resolution, anti-aliasing, and compression quality.
-   **Headless Operation**: Runs from the command line without needing a graphical interface, making it perfect for server-side automation.

## Installation

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
It is highly recommended to use a Python virtual environment.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required libraries
pip install -r requirements.txt
```

**Notes:**
-   **For WebP Support**: To enable saving images in the `.webp` format, you may need to install additional libraries for Pillow. You can do this by running:
    ```bash
    pip install Pillow[libwebp]
    ```
-   **For Headless Linux**: PyVista may require an off-screen framebuffer to run without a display. You can install it on Debian/Ubuntu with:
    ```bash
    sudo apt-get install libgl1-mesa-glx xvfb
    ```

---

## Usage

The script can be run from your terminal. Point it to a single file or a directory of 3D models.

### Basic Examples

**1. Process a single STL file:**
```bash
python parallel.py path/to/your/model.stl
```

**2. Process an entire directory:**
This will find all `.stl` and `.3mf` files in the `models` directory and save the images to the `output_previews` directory.
```bash
python parallel.py ./models/ --output ./output_previews/
```

### Advanced Examples

**1. Decimate and Optimize:**
Process a directory, decimating each mesh by 90% and saving the output as high-quality JPEGs.
```bash
python parallel.py ./models/ -o ./output_previews/ --decimate 0.9 --format jpg --quality 85
```

**2. High-Resolution PNG with Quantization:**
Generate high-resolution images, disabling anti-aliasing for a sharper look, and reducing the PNG file size by limiting it to 128 colors.
```bash
python parallel.py ./models/ -o ./output_previews/ --resolution 1920 1080 --no-aa --format png --quality 128```

**3. Sequential (Single-Core) Processing:**
To process files one by one, which can be useful for debugging or on memory-constrained systems, use the `--no-parallel` flag.
```bash
python parallel.py ./models/ --no-parallel
```

---

## Command-Line Arguments

| Argument | Short Form | Description | Default |
|---|---|---|---|
| `input_path` | | Path to a single `.stl`/`.3mf` file or a directory containing them. | (Required) |
| `--output` | `-o` | Output directory for the generated images. | `.` (current directory) |
| `--decimate` | `-d` | Mesh decimation factor (e.g., 0.9 for 90% reduction). | `0.0` (disabled) |
| `--resolution`| `-r` | Output image resolution in `Width Height`. | `800 800` |
| `--no-aa` | | Disables anti-aliasing for faster, sharper rendering. | (Flag, AA is on by default) |
| `--format` | | The output image format. | `png` |
| `--quality` | | Compression quality. For JPG/WebP: 1-100. For PNG: number of colors (2-256). | `0` (lossless) |
| `--no-parallel`| | Disables parallel processing and runs sequentially. | (Flag, parallel is on by default) |

---

## License

This project is licensed under the MIT License.
