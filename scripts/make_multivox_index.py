"""Script to create the MULTIVOX dataset index.

This script generates an index file for the MULTIVOX dataset by:
1. Reading metadata.csv to get all track IDs
2. Finding associated audio/video files for each track
3. Computing MD5 checksums for all files
4. Creating the index structure

Note: This script assumes a specific file structure. If the actual structure
differs, the file path patterns will need to be adjusted.
"""

import argparse
import csv
import glob
import hashlib
import json
import os


def md5(file_path):
    """Get md5 hash of a file.

    Args:
        file_path (str): File path

    Returns:
        str: md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Get the absolute path to the index file based on this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_INDEX_PATH = os.path.join(
    PROJECT_ROOT, "mirdata", "datasets", "indexes", "mirdata_multivox_index_1.0.json"
)


def find_nearfield_files(base_path, singer_ids, track_dir):
    """Find near-field audio files for a track.

    Args:
        base_path (str): Base path to the dataset
        singer_ids (list): List of potential singer IDs (e.g., ["S1", "S2", "Sinstructor"])
        track_dir (str): Path to the track directory (already resolved)

    Returns:
        dict: Dictionary mapping singer ID to (path, checksum) tuples
    """
    if not os.path.exists(track_dir):
        return {}

    nearfield_files = {}

    # Match near-field files to singer IDs
    # Pattern: {SINGER_ID}_*.wav (e.g., S1_*.wav, S10_*.wav, Sinstructor_*.wav)
    # These are all .wav files that are not ORTF L or R files
    for singer_id in singer_ids:
        if not singer_id:
            continue
        pattern = os.path.join(track_dir, f"{singer_id}_*.wav")
        # Prefer deterministic order; if multiple variants exist, pick the longest
        # basename (more specific) and then lexicographically.
        matches = sorted(
            glob.glob(pattern),
            key=lambda p: (len(os.path.basename(p)), os.path.basename(p)),
        )
        if matches:
            nearfield_file = matches[-1]  # longest/specific, then alpha
            rel_path = os.path.relpath(nearfield_file, base_path)
            checksum = md5(nearfield_file)
            nearfield_files[singer_id] = (rel_path, checksum)

    return nearfield_files


def make_multivox_index(data_path):
    """Create the MULTIVOX dataset index.

    Args:
        data_path (str): Path to the MULTIVOX dataset root directory
    """
    metadata_path = os.path.join(data_path, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.csv not found at {metadata_path}. "
            "Please ensure the dataset is downloaded and metadata.csv is in the root directory."
        )

    # Read metadata to get all track information
    tracks_metadata = {}
    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row["Path"].strip()
            tracks_metadata[track_id] = row

    print(f"Found {len(tracks_metadata)} tracks in metadata.csv")

    # File naming patterns (using pattern matching, not exact filenames):
    # - ORTF L: audio_ORTF_L*.wav (pattern match)
    # - ORTF R: audio_ORTF_R*.wav (pattern match)
    # - Video: video360*.mp4 (pattern match)
    # - Audio 360: audio360*.wav (pattern match) - stereo audio from 360° camera
    # - Near-field: {SINGER_ID}_*.wav (pattern match, all .wav files except ORTF L/R/audio360)

    index = {
        "version": "1.0",
        "tracks": {},
        "metadata": {
            "metadata_file": (
                "metadata.csv",
                md5(metadata_path),
            ),
        },
    }

    # Process each track
    for track_id, metadata in tracks_metadata.items():
        print(f"Processing track: {track_id}")

        track_entry = {}

        # Get condition from CSV metadata
        condition = metadata.get("Condition", "").strip()  # C1, C2, C3, or C4

        # Try both: direct path and inside condition directory
        track_dir = os.path.join(data_path, track_id)
        if not os.path.exists(track_dir):
            track_dir = os.path.join(data_path, condition, track_id)

        if os.path.exists(track_dir):
            # Find ORTF L file: pattern matching for audio_ORTF_L*.wav
            ortf_l_matches = sorted(
                glob.glob(os.path.join(track_dir, "audio_ORTF_L*.wav"))
            )
            if ortf_l_matches:
                ortf_l_file = ortf_l_matches[0]
                rel_path = os.path.relpath(ortf_l_file, data_path)
                track_entry["audio_ortf_l"] = (rel_path, md5(ortf_l_file))

            # Find ORTF R file: pattern matching for audio_ORTF_R*.wav
            ortf_r_matches = sorted(
                glob.glob(os.path.join(track_dir, "audio_ORTF_R*.wav"))
            )
            if ortf_r_matches:
                ortf_r_file = ortf_r_matches[0]
                rel_path = os.path.relpath(ortf_r_file, data_path)
                track_entry["audio_ortf_r"] = (rel_path, md5(ortf_r_file))

            # Find 360° video: just find video360*.mp4 file
            video_matches = sorted(glob.glob(os.path.join(track_dir, "video360*.mp4")))
            if video_matches:
                video_file = video_matches[0]
                rel_path = os.path.relpath(video_file, data_path)
                track_entry["video_360"] = (rel_path, md5(video_file))

            # Find 360° camera audio: audio360*.wav file (stereo audio from Insta360 X3 camera)
            audio360_matches = sorted(glob.glob(os.path.join(track_dir, "audio360*.wav")))
            if audio360_matches:
                audio360_file = audio360_matches[0]
                rel_path = os.path.relpath(audio360_file, data_path)
                track_entry["audio_360"] = (rel_path, md5(audio360_file))

            # Find near-field recordings: all remaining .wav files that are not ORTF L or R or audio360
            # We look for all circle positions plus any IDs listed in Nearfield_Files_Captured
            locations_at_circle_str = metadata.get("Locations_At_Circle", "").strip()
            locations_at_circle = (
                int(locations_at_circle_str) if locations_at_circle_str.isdigit() else 0
            )
            circle_ids = [f"S{i+1}" for i in range(locations_at_circle)]

            nearfield_list_str = metadata.get("Nearfield_Files_Captured", "").strip()
            captured_ids = (
                [s.strip() for s in nearfield_list_str.split(",")]
                if nearfield_list_str
                else []
            )

            # Combine IDs, preserving order but removing duplicates
            all_potential_ids = list(dict.fromkeys(circle_ids + captured_ids))

            if all_potential_ids:
                nearfield_files = find_nearfield_files(
                    data_path, all_potential_ids, track_dir
                )
                if nearfield_files:
                    # Store as flat keys (near_field_S1, near_field_S2, etc.)
                    # to adhere to mirdata validation conventions
                    for singer_id, (path, checksum) in nearfield_files.items():
                        track_entry[f"near_field_{singer_id}"] = (path, checksum)
        else:
            print(f"Warning: Track directory not found: {track_dir}")

        index["tracks"][track_id] = track_entry

    # Save index
    os.makedirs(os.path.dirname(DATASET_INDEX_PATH), exist_ok=True)
    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)

    print(f"Index created successfully at {DATASET_INDEX_PATH}")
    print(f"Total tracks indexed: {len(index['tracks'])}")


def main(args):
    make_multivox_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make MULTIVOX dataset index file.")
    PARSER.add_argument(
        "data_path",
        type=str,
        help="Path to MULTIVOX dataset root directory (should contain metadata.csv)",
    )

    main(PARSER.parse_args())
