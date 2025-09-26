import os
import argparse
from PIL import Image

def resize_images(input_folder, output_folder, size=(256, 256)):
    """
    Resize all images in input_folder to the given size
    and save them in output_folder.
    
    Args:
        input_folder (str): Path to the folder containing input images
        output_folder (str): Path to the folder where resized images will be saved
        size (tuple): Desired size (width, height)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            img_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(img_path)
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
                
                save_path = os.path.join(output_folder, filename)
                img_resized.save(save_path)
                print(f"Resized and saved: {save_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a folder.")
    parser.add_argument("--dir", required=True, type=str,
                        help="The folder that contains the images to be resized")
    parser.add_argument("--size", required=True, nargs=2, type=int,
                        metavar=("WIDTH", "HEIGHT"),
                        help="The final size of the images (width height)")
    args = parser.parse_args()
    input_folder = args.dir      # change this to your folder
    size = tuple(args.size)
    output_folder = input_folder   # output folder
    # size = (360, 640)                  # specify desired size here
    
    resize_images(input_folder, output_folder, size)
