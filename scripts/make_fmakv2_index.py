import argparse
import glob
import json
import os
import zipfile
import pandas as pd
from mirdata.validate import md5

DATASET_INDEX_PATH = os.path.abspath("../mirdata/datasets/indexes/fmakv2_index_1.0.json")

import os
import zipfile
import glob

import os
import zipfile
import glob
import shutil


def extract_audio_files(dataset_data_path):
    audio_dir = os.path.join(dataset_data_path, "audio")
    archives_dir = os.path.join(dataset_data_path, "archives")
    temp_extract_dir = os.path.join(dataset_data_path, "temp_extracted")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(temp_extract_dir, exist_ok=True)

    zip_files = glob.glob(os.path.join(archives_dir, "*.zip"))

    if not zip_files:
        print("No .zip files found in /archives")
        return

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(temp_extract_dir)
            print(f"Extracted {zip_file} into temp_extracted/")

    sub_zip_files = glob.glob(os.path.join(temp_extract_dir, "*.zip"))

    for sub_zip in sub_zip_files:
        with zipfile.ZipFile(sub_zip, "r") as z:
            z.extractall(temp_extract_dir)
            print(f"Extracted {sub_zip}")

    for root, _, files in os.walk(temp_extract_dir):
        for file in files:
            if file.endswith(".mp3"):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(audio_dir, file)

                if not os.path.exists(dest_path):
                    shutil.move(src_path, dest_path)
                    print(f"Moved {file} to audio/")

    print("finished extracting and moving audio files.")

def make_dataset_index(dataset_data_path):
    metadata_path = os.path.join(dataset_data_path, "metadata", "metadata_fmakv2.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"file at {metadata_path} not found")

    extract_audio_files(dataset_data_path)
    metadata_checksum = md5(metadata_path)
    df = pd.read_csv(metadata_path)

    index_tracks = {}
    audio_dir = os.path.join(dataset_data_path, "audio")

    for _, row in df.iterrows():
        track_id = str(row["track_id"]).strip()
        key_and_mode = row["key_and_mode"]
        spotify_uri = row["spotify_uri"]
        key_and_mode = None if pd.isna(key_and_mode) else key_and_mode
        spotify_uri = None if pd.isna(spotify_uri) else spotify_uri

        # finding matching mp3
        audio_files = glob.glob(os.path.join(audio_dir, f"{track_id}.*"))
        if not audio_files:
            print(f"### mp3 not found for track_id {track_id}")
            continue

        audio_file = audio_files[0]
        audio_checksum = md5(audio_file)

        index_tracks[track_id] = {
            "audio": (os.path.basename(audio_file), audio_checksum),
            "key_and_mode": key_and_mode,
            "spotify_uri": spotify_uri,
        }
    # final index
    dataset_index = {
        "version": "1.0",
        "metadata": {"metadata_fmakv2.csv": ("metadata/metadata_fmakv2.csv", metadata_checksum)},
        "tracks": index_tracks,
    }

    os.makedirs(os.path.dirname(DATASET_INDEX_PATH), exist_ok=True)
    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)

    print(f"json index generated: {DATASET_INDEX_PATH}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument("dataset_data_path", type=str, help="Path to dataset folder.")
    args = PARSER.parse_args()
    make_dataset_index(args.dataset_data_path)