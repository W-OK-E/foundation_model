import os
import shutil

def copy_and_rename(
    src_root,
    dst_dir,
    splits=("train", "test", "val"),
    ext=".png"
):
    os.makedirs(dst_dir, exist_ok=True)

    idx = 0
    for split in splits:
        split_dir = os.path.join(src_root, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping missing folder: {split_dir}")
            continue

        files = sorted(f for f in os.listdir(split_dir) if f.endswith(ext))

        for fname in files:
            src_path = os.path.join(split_dir, fname)
            dst_name = f"{idx:04d}{ext}"
            dst_path = os.path.join(dst_dir, dst_name)

            shutil.copy2(src_path, dst_path)
            idx += 1

    print(f"Copied {idx} files to {dst_dir}")
