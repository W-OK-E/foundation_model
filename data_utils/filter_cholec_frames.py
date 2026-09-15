#!/usr/bin/env python3
"""
Script to filter frames from Cholec80 dataset based on empty Triplets data.

This script reads the Cholec80_KG_annotations.xlsx file and identifies frames
where the Triplets column contains only '{}'. These frames are excluded from
consideration. The script generates a list of valid frame IDs for each video.
"""

import pandas as pd
import os
import json
from pathlib import Path


def load_annotations(excel_path):
    """Load all sheets from the Excel annotations file."""
    return pd.ExcelFile(excel_path)


def get_video_id_from_sheet(sheet_name):
    """Extract video ID from sheet name (e.g., 'SheetVID01' -> 'VID01')."""
    return sheet_name.replace("Sheet", "")


def filter_frames_for_video(xl_file, sheet_name):
    """
    Filter frames for a single video, keeping only frames with non-empty Triplets.

    Args:
        xl_file: ExcelFile object
        sheet_name: Name of the sheet to process

    Returns:
        tuple: (video_id, list of valid frame_ids)
    """
    df = pd.read_excel(xl_file, sheet_name=sheet_name)

    # Check if Triplets column exists
    if "Triplets" not in df.columns:
        print(
            f"Warning: 'Triplets' column not found in {sheet_name}, skipping all frames"
        )
        video_id = get_video_id_from_sheet(sheet_name)
        return video_id, []

    # Filter out frames where Triplets is exactly '{}'
    valid_frames = df[df["Triplets"] != "{}"]["Frame ID"].tolist()

    video_id = get_video_id_from_sheet(sheet_name)

    return video_id, valid_frames


def process_all_videos(excel_path, video_data_path):
    """
    Process all videos and generate valid frame lists.

    Args:
        excel_path: Path to the Excel annotations file
        video_data_path: Path to the video data directory

    Returns:
        dict: Dictionary mapping video_id to list of valid frame_ids
    """
    # Load annotations
    xl_file = load_annotations(excel_path)

    # Get all sheet names that correspond to videos (skip the first 'Sheet')
    video_sheets = [
        sheet for sheet in xl_file.sheet_names if sheet.startswith("SheetVID")
    ]

    valid_frames_dict = {}

    for sheet_name in video_sheets:
        video_id, valid_frames = filter_frames_for_video(xl_file, sheet_name)

        # Check if video folder exists
        video_folder = os.path.join(video_data_path, video_id)
        if os.path.exists(video_folder):
            valid_frames_dict[video_id] = valid_frames
            print(
                f"Processed {video_id}: {len(valid_frames)} valid frames out of {len(pd.read_excel(xl_file, sheet_name=sheet_name))} total frames"
            )
        else:
            print(
                f"Warning: Video folder {video_folder} not found, skipping {video_id}"
            )

    return valid_frames_dict


def save_results(valid_frames_dict, output_path):
    """Save the results to a JSON file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(valid_frames_dict, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    # Define paths
    excel_path = "/mnt/data/omkumar/mounted_datasets/Cholec80_KG_annotations.xlsx"
    video_data_path = (
        "/mnt/data/omkumar/mounted_datasets/Surgical/Cholec80/cholec80/data"
    )
    output_path = "/mnt/data/omkumar/foundation_model/foundation_phase1/data_utils/valid_frames.json"

    # Check if input files exist
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    if not os.path.exists(video_data_path):
        raise FileNotFoundError(f"Video data directory not found: {video_data_path}")

    print("Processing Cholec80 annotations to filter frames...")
    print(f"Excel file: {excel_path}")
    print(f"Video data path: {video_data_path}")
    print()

    # Process all videos
    valid_frames_dict = process_all_videos(excel_path, video_data_path)

    # Save results
    save_results(valid_frames_dict, output_path)

    # Print summary
    total_frames = sum(len(frames) for frames in valid_frames_dict.values())
    total_videos = len(valid_frames_dict)

    print()
    print(f"Summary:")
    print(f"- Processed {total_videos} videos")
    print(f"- Total valid frames: {total_frames}")
    print(f"- Average frames per video: {total_frames / total_videos:.1f}")


if __name__ == "__main__":
    main()
