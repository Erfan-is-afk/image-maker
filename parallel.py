# --- START OF FILE compress_fast.py ---

import os
import sys
import argparse
import time
import math # Needed for grid packing
import numpy as np
import pyvista as pv
from tqdm import tqdm

# --- NEW: Import multiprocessing library ---
import multiprocessing

# --- DEPENDENCY IMPORTS ---
try:
    import trimesh
except ImportError:
    print("\n--- CRITICAL DEPENDENCY MISSING ---")
    print("Error: The 'trimesh' library is required to run this script.")
    print("Please install it by running: pip install trimesh lxml")
    sys.exit(1)

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


def optimize_image(file_path, quality, new_format):
    # This function remains unchanged
    if not PILLOW_AVAILABLE:
        tqdm.write("[!] Cannot optimize: Pillow library not installed. (pip install Pillow)")
        return None, None, file_path

    try:
        original_size = os.path.getsize(file_path)
        img = Image.open(file_path).convert('RGBA')

        base, _ = os.path.splitext(file_path)
        new_path = f"{base}.{new_format}"
        save_options = {}
        img_to_save = img

        if new_format == 'jpg':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img_to_save = background
            save_options = {'quality': quality, 'optimize': True}
        elif new_format == 'webp':
            save_options = {'quality': quality, 'method': 6}
        elif new_format == 'png' and quality > 0:
            quant_method = Image.Quantize.LIBIMAGEQUANT if hasattr(Image.Quantize, 'LIBIMAGEQUANT') else Image.Quantize.FASTOCTREE
            img_to_save = img.quantize(colors=quality, method=quant_method)
        else: # Default lossless PNG
            save_options = {'optimize': True}

        img_to_save.save(new_path, **save_options)

        if new_path != file_path:
            os.remove(file_path)

        new_size = os.path.getsize(new_path)
        return original_size, new_size, new_path

    except Exception as e:
        tqdm.write(f"\n[!] Could not optimize {os.path.basename(file_path)}: {e}")
        if new_format == 'webp' and 'encoder' in str(e).lower():
            tqdm.write("[!] Hint: For WebP support, you may need to install webp libraries for Pillow.")
            tqdm.write("[!] Try: pip install Pillow[libwebp]")
        return None, None, file_path


def generate_fast_iso_view(
    input_path,
    output_dir,
    resolution,
    decimation_target,
    use_aa
):
    try:
        loaded_data = trimesh.load(input_path, force='scene')
        geometries = list(loaded_data.geometry.values()) if isinstance(loaded_data, trimesh.Scene) else [loaded_data]

        if not geometries:
            raise ValueError("File contains no 3D geometry.")

        if len(geometries) > 1:
            tqdm.write(f"--> Multi-part file detected ({len(geometries)} objects). Arranging to prevent occlusion.")
            geometries.sort(key=lambda g: g.bounding_box.extents[2], reverse=True)
            num_objects = len(geometries)
            num_cols = math.ceil(math.sqrt(num_objects))
            num_rows = math.ceil(num_objects / num_cols)
            avg_dim = sum(max(g.extents) for g in geometries) / num_objects
            spacing = avg_dim * 0.2
            col_widths = [0.0] * num_cols
            row_heights = [0.0] * num_rows
            for i, geom in enumerate(geometries):
                row, col = i // num_cols, i % num_cols
                extents = geom.bounding_box.extents
                col_widths[col] = max(col_widths[col], extents[0])
                row_heights[row] = max(row_heights[row], extents[1])
            x_offsets = [0.0] * num_cols
            y_offsets = [0.0] * num_rows
            for i in range(1, num_cols):
                x_offsets[i] = x_offsets[i-1] + col_widths[i-1] + spacing
            for i in range(1, num_rows):
                y_offsets[i] = y_offsets[i-1] + row_heights[i-1] + spacing
            repositioned_geometries = []
            for i, geom in enumerate(geometries):
                row, col = i // num_cols, i % num_cols
                min_coords, _ = geom.bounds
                translation_vector = [x_offsets[col] - min_coords[0], y_offsets[row] - min_coords[1], -min_coords[2]]
                geom.apply_translation(translation_vector)
                repositioned_geometries.append(geom)
            trimesh_mesh = trimesh.util.concatenate(repositioned_geometries)
        else:
            trimesh_mesh = geometries[0]
        mesh = pv.wrap(trimesh_mesh)
        if mesh.n_points == 0:
            raise ValueError("Loaded mesh is empty after processing.")
    except Exception as e:
        tqdm.write(f"\n[!] Failed to load and process {os.path.basename(input_path)}: {e}")
        return None

    if decimation_target > 0.0:
        tqdm.write("--> Starting mesh decimation...")
        try:
            mesh = mesh.decimate(target_reduction=decimation_target)
        except Exception as e:
            tqdm.write(f"\n[!] Could not decimate {os.path.basename(input_path)}: {e}")
    
    unit_label, label_format = "mm", "%.0f"
    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    plotter.add_mesh(mesh, color='gainsboro', smooth_shading=False)
    original_bounds = mesh.bounds
    x_len, y_len, z_len = original_bounds[1] - original_bounds[0], original_bounds[3] - original_bounds[2], original_bounds[5] - original_bounds[4]
    max_dim = max(x_len, y_len, z_len)
    padding = max_dim * 0.25 
    translation_vector = [-original_bounds[0], -original_bounds[2], -original_bounds[4]]
    mesh.translate(translation_vector, inplace=True)
    padded_bounds = [0 - padding, x_len + padding, 0 - padding, y_len + padding, 0 - padding, z_len + padding]
    plotter.camera_position = 'iso'
    plotter.reset_camera(bounds=padded_bounds)
    plotter.enable_parallel_projection()
    if use_aa:
        plotter.enable_anti_aliasing()
    plotter.show_bounds(
        grid='back', location='outer', ticks='both',
        xtitle=f'X ({unit_label})', ytitle=f'Y ({unit_label})', ztitle=f'Z ({unit_label})',
        fmt=label_format, font_size=16, color='black', n_xlabels=3, n_ylabels=3, n_zlabels=3
    )
    plotter.set_background('white', top='lightgrey')
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = os.path.join(output_dir, f"{base_name}_iso.png")
    try:
        plotter.screenshot(output_filename, transparent_background=True)
    except Exception as e:
        tqdm.write(f"\n[!] Failed to save screenshot for {os.path.basename(input_path)}: {e}")
        return None
    finally:
        plotter.close()
    return output_filename

# This function encapsulates the logic for a single file.
def process_single_file(file_path, args):
    """Worker function that processes one file and returns True on success."""
    try:
        file_start_time = time.perf_counter()
        
        output_file = generate_fast_iso_view(
            input_path=file_path,
            output_dir=args.output,
            resolution=tuple(args.resolution),
            decimation_target=args.decimate,
            use_aa=args.use_aa
        )
        
        if not output_file:
            # generate_fast_iso_view already printed an error
            return False

        elapsed_time = time.perf_counter() - file_start_time
        tqdm.write(f"-> Generated initial '{os.path.basename(output_file)}' in {elapsed_time:.2f}s")
        
        if args.quality > 0:
            orig_size, new_size, new_path = optimize_image(output_file, args.quality, args.format)
            if orig_size and new_size:
                reduction_pct = (1 - new_size / orig_size) * 100
                tqdm.write(f"   Optimized '{os.path.basename(new_path)}': {orig_size/1024:.1f}KB -> {new_size/1024:.1f}KB (saved {reduction_pct:.1f}%)")
        
        return True # Indicate success
    except Exception as e:
        tqdm.write(f"\n[!!!] An unexpected error occurred while processing {os.path.basename(file_path)}: {e}")
        return False # Indicate failure


def main():
    parser = argparse.ArgumentParser(
        description="Generate fast isometric views of 3D models (STL/3MF).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_path', help="Path to a single .stl/.3mf file or a directory.")
    parser.add_argument('--output', '-o', default='.', help="Output directory. (Default: current)")
    parser.add_argument('--decimate', '-d', type=float, default=0.0, help="Mesh decimation target (0.0 to 0.95).")
    parser.add_argument('--resolution', '-r', type=int, nargs=2, default=[800, 800], metavar=('W', 'H'), help="Output image resolution.")
    parser.add_argument('--no-aa', action='store_false', dest='use_aa', help="Disable anti-aliasing.")
    parser.add_argument('--format', choices=['png', 'jpg', 'webp'], default='png', help="Output image format.")
    parser.add_argument('--quality', type=int, default=0, help="Compression quality (1-100 for jpg/webp, 2-256 for png colors).")
    parser.add_argument('--no-parallel', action='store_true', help="Disable parallel processing and run sequentially.")
    
    args = parser.parse_args()
    
    if args.quality > 0 and not PILLOW_AVAILABLE:
        print("\n--- WARNING: CANNOT OPTIMIZE ---")
        print("Continuing without extra compression...\n")
        
    os.makedirs(args.output, exist_ok=True)
    files_to_process = []
    if os.path.isdir(args.input_path):
        for filename in sorted(os.listdir(args.input_path)):
            if filename.lower().endswith(('.stl', '.3mf')):
                files_to_process.append(os.path.join(args.input_path, filename))
    elif os.path.isfile(args.input_path):
        files_to_process.append(args.input_path)
    
    if not files_to_process:
        print("No .stl or .3mf files found.")
        return

    print("\n--- Processing Settings ---")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"Anti-Aliasing: {'Enabled' if args.use_aa else 'Disabled (FAST)'}")
    print(f"Decimation: {args.decimate * 100:.0f}% reduction")
    print(f"Output Format: {args.format.upper()}")
    if args.quality > 0:
        if args.format in ['jpg', 'webp']: print(f"Quality: {args.quality}")
        else: print(f"Quality: {args.quality} Colors (Quantization)")
    else: print("Quality: Lossless")
    print(f"Outputting to: {os.path.abspath(args.output)}")
    print("-" * 27)

    total_start_time = time.perf_counter()
    
    # --- MODIFIED: Main processing loop now handles both parallel and sequential ---
    run_in_parallel = not args.no_parallel and len(files_to_process) > 1

    if run_in_parallel:
        # Use one less than the total number of CPUs to keep the system responsive
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"\nStarting parallel processing with {num_processes} cores...")
        
        # Prepare arguments for each worker process
        tasks = [(f, args) for f in files_to_process]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use starmap to pass multiple arguments to the worker
            # Wrap the pool iterator with tqdm for a progress bar
            results = list(tqdm(pool.starmap(process_single_file, tasks), total=len(tasks), desc="Generating Images"))
        
        success_count = sum(1 for r in results if r is True)
    else:
        # Fallback to the original sequential loop
        if len(files_to_process) > 1:
            print("\nProcessing files sequentially...")
        success_count = 0
        for f in tqdm(files_to_process, desc="Generating Images"):
            if process_single_file(f, args):
                success_count += 1
            
    total_end_time = time.perf_counter()
    total_elapsed_time = total_end_time - total_start_time
    
    print("\n--- Processing Complete ---")
    print(f"Total files processed: {len(files_to_process)}")
    print(f"Successfully generated: {success_count}")
    print(f"Total time elapsed: {total_elapsed_time:.2f} seconds.")
    if success_count > 0:
        avg_time = total_elapsed_time / success_count
        print(f"Average time per image: {avg_time:.2f} seconds.")
    print("---------------------------\n")


if __name__ == "__main__":
    # This check is crucial for multiprocessing to work correctly on all platforms
    main()