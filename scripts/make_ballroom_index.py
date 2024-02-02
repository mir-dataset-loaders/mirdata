import argparse
import hashlib
import json
import os
import glob
from mirdata.validate import md5

BALLROOM_RHYTHM_INDEX_PATH = "../mirdata/datasets/indexes/ballroom_full_index_1.0.json"

def make_ballroom_rhythm_index(dataset_data_path):
    cmr_index = {
        "version": "1.0",
        "tracks": {},
    }
    
    dataset_folder_name = "B_1.0"  # Update this folder name if needed
    for root, dirs, files in os.walk(dataset_data_path):
        for filename in files:
            if filename.endswith(".wav"):
                subfolder = os.path.basename(root)
                idx = filename.split(".")[0]
                cmr_index["tracks"][idx] = {
                    "audio": (
                        os.path.join(dataset_folder_name, "audio", subfolder, filename),
                        md5(os.path.join(root, filename)),
                    ),
                    "beats": (
                        os.path.join(dataset_folder_name, "annotations", "beats", filename.replace(".wav", ".beats")),
                        md5(os.path.join(dataset_data_path, "annotations", "beats",  filename.replace(".wav", ".beats"))),
                    ),
                    "tempo": (
                        os.path.join(dataset_folder_name, "annotations", "tempo",  filename.replace(".wav", ".bpm")),
                        md5(os.path.join(dataset_data_path, "annotations", "tempo",  filename.replace(".wav", ".bpm"))),
                    )
                }

    with open(BALLROOM_RHYTHM_INDEX_PATH, "w") as fhandle:
        json.dump(cmr_index, fhandle, indent=2)

def main(args):
    print("creating index...")
    make_ballroom_rhythm_index(args.dataset_data_path)
    print("done!")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Ballroom Rhythm index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to Ballroom Rhythm data folder."
    )

    main(PARSER.parse_args())
