import os
import json
import argparse

def create_split_json(folder_path, output_file="split.json"):
    splits = {}

    # Loop through all .txt files
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            split_name = os.path.splitext(file_name)[0]  # e.g., "train" from "train.txt"
            
            if(split_name == "viz" or split_name == "image_shape.txt"):
                continue

            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]  # remove empty lines
            if(split_name == "test"):
                splits["viz"] = lines[:8]

            splits[split_name] = lines

    # Save as JSON
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"✅ Split file created at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create split.json from txt files")
    parser.add_argument("--folder", type=str, help="Path to folder containing .txt files")
    args = parser.parse_args()

    create_split_json(args.folder)
