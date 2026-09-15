import os
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def save_frame(frame, path):
    """Helper function to save a single frame."""
    cv2.imwrite(path, frame)

def extract_frames_from_videos(videos_dir, annotations_dir, output_dir):
    """
    For each subfolder in annotations_dir, count the annotation images,
    extract that many frames from the corresponding video in videos_dir,
    and save them under output_dir preserving subfolder structure.
    The output frames have the same filenames as the annotations.
    """
    os.makedirs(output_dir, exist_ok=True)

    for subfolder in sorted(os.listdir(annotations_dir)):
        output_subdir = os.path.join(output_dir, subfolder)

        # Skip if already processed
        if os.path.exists(output_subdir):
            print(f'{output_subdir} exists, skipping')
            continue

        subfolder_path = os.path.join(annotations_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Get annotation filenames
        image_files = sorted([
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        num_annotations = len(image_files)
        if num_annotations == 0:
            print(f"⚠️  No annotation images found in {subfolder_path}, skipping.")
            continue

        # Locate corresponding video
        video_path = os.path.join(videos_dir, f"{subfolder}.mp4")
        if not os.path.exists(video_path):
            print(f"❌ Video not found for {subfolder} -> {video_path}")
            continue

        # Open video and read metadata
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or fps_video == 0:
            print(f"⚠️  Could not read video metadata for {video_path}")
            cap.release()
            continue

        # Determine frame interval to match annotation count
        frame_interval = max(int(total_frames / num_annotations), 1)
        os.makedirs(output_subdir, exist_ok=True)

        print(f"📹 Processing {video_path}")
        print(f"   Total frames: {total_frames}, Annotations: {num_annotations}, Interval: {frame_interval}")

        frame_count = 0
        saved_frames = 0
        futures = []
rsync -avzhe "ssh -T -c aes128-gcm@openssh.com -o Compression=no -x" --progress --inplace --whole-file --stats /mnt/data/omkumar/foundation_phase1/datasets/cataract1k asavari@10.9.7.15:/home/asavari/Cataracts

        with ThreadPoolExecutor(max_workers=20) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frames evenly
                if frame_count % frame_interval == 0 and saved_frames < num_annotations:
                    # Use the same filename as the annotation image
                    frame_path = os.path.join(output_subdir, f'{saved_frames:04d}.png')

                    futures.append(executor.submit(save_frame, frame.copy(), frame_path))
                    saved_frames += 1

                    # Prevent memory buildup
                    if len(futures) > 1000:
                        for f in as_completed(futures):
                            pass
                        futures.clear()

                frame_count += 1
                if saved_frames >= num_annotations:
                    break

            # Wait for all remaining tasks
            for f in as_completed(futures):
                pass

        cap.release()
        print(f"✅ Saved {saved_frames} frames to {output_subdir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos to match annotation count.")
    parser.add_argument("--videos_dir", required=True, help="Path to folder containing cataract_1k_videos")
    parser.add_argument("--annotations_dir", required=True, help="Path to folder containing annotation subfolders")
    parser.add_argument("--output_dir", required=True, help="Path where extracted images should be saved")

    args = parser.parse_args()
    extract_frames_from_videos(args.videos_dir, args.annotations_dir, args.output_dir)
