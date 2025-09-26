import os

# paths to your folders
folder1 = "path/to/folder1"
folder2 = "path/to/folder2"

# get sorted lists of files (only images)
files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# safety check
if len(files1) != len(files2):
    raise ValueError(f"Folders have different number of files: {len(files1)} vs {len(files2)}")

# rename files in folder2 to match folder1
for f1, f2 in zip(files1, files2):
    src = os.path.join(folder2, f2)
    dst = os.path.join(folder2, f1)

    # if dst already exists, add a temporary rename to avoid overwriting
    if os.path.exists(dst):
        os.rename(src, dst + ".tmp")
    else:
        os.rename(src, dst)

print("Renaming completed successfully.")
