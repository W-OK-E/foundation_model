import os
import argparse
from PIL import Image

def grayscale_to_rgb(input_folder, output_folder, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    """
    Convert grayscale images to RGB and save them.
    
    Args:
        input_folder (str): Path to the folder containing grayscale images.
        output_folder (str): Path to the folder where RGB images will be saved.
        exts (tuple): File extensions to process.
    """
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(exts):
            continue

        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        try:
            img = Image.open(in_path).convert("L")   # Force grayscale
            rgb_img = img.convert("RGB")            # Convert to RGB
            rgb_img.save(out_path)
            print(f"Converted: {fname}")
        except Exception as e:
            print(f"⚠️ Failed to process {fname}: {e}")

    print("\n✅ Conversion complete.")

# Argument parsing
parser = argparse.ArgumentParser(description="Randomly split dataset into train, val, and test sets.")
parser.add_argument("--dir", type=str, required=True, help="Directory containing the dataset files.")
args = parser.parse_args()

if __name__ == "__main__":
    input_folder = args.dir
    output_folder = input_folder
    grayscale_to_rgb(input_folder, output_folder)
