import argparse
import hashlib
import json
import os
from mirdata.validate import md5

BRID_RHYTHM_INDEX_PATH = "../mirdata/datasets/indexes/brid_full_index_1.0.json"

def make_brid_rhythm_index(dataset_data_path):
    cmr_index = {
        "version": "1.0",
        "tracks": {},
    }
    
    dataset_folder_name = "BRID_1.0"  # Adjust as needed for your dataset's root directory in mirdata
    
    # Define paths for beats and tempo annotations
    beats_path = os.path.join(dataset_data_path, "Annotations", "beats")
    tempo_path = os.path.join(dataset_data_path, "Annotations", "tempo")
    
    for root, dirs, files in os.walk(dataset_data_path):
        for filename in files:
            if filename.endswith(".wav"):
                # Extract relevant path details
                idx = filename.split(".")[0]
                relative_audio_path = os.path.relpath(os.path.join(root, filename), dataset_data_path)
                
                # Construct paths for annotations
                beat_file = f"{idx}.beats"
                tempo_file = f"{idx}.bpm"
                
                # Set relative paths to None if files do not exist
                relative_beat_path = os.path.join("Annotations", "beats", beat_file) if os.path.exists(os.path.join(beats_path, beat_file)) else None
                relative_tempo_path = os.path.join("Annotations", "tempo", tempo_file) if os.path.exists(os.path.join(tempo_path, tempo_file)) else None

                
                # Check if annotation files exist
                beat_file_path = os.path.join(beats_path, beat_file) if relative_beat_path else None
                tempo_file_path = os.path.join(tempo_path, tempo_file) if relative_tempo_path else None

                # Add track information to index
                cmr_index["tracks"][idx] = {
                    "audio": (
                        os.path.join(dataset_folder_name, relative_audio_path),
                        md5(os.path.join(root, filename)),
                    ),
                    "beats": (
                        os.path.join(dataset_folder_name, relative_beat_path) if relative_beat_path else None,
                        md5(beat_file_path) if beat_file_path else None,
                    ),
                    "tempo": (
                        os.path.join(dataset_folder_name, relative_tempo_path) if relative_tempo_path else None,
                        md5(tempo_file_path) if tempo_file_path else None,
                    )
                }

    # Write to index file
    with open(BRID_RHYTHM_INDEX_PATH, "w") as fhandle:
        json.dump(cmr_index, fhandle, indent=2)

def main(args):
    print("Creating index...")
    make_brid_rhythm_index(args.dataset_data_path)
    print("Index creation done!")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make BRID Rhythm index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to the BRID Rhythm data folder."
    )
    main(PARSER.parse_args())
