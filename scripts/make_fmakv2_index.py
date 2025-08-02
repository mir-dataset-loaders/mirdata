import argparse
import csv
import json
import os
from mirdata.validate import md5

FMAKV2_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mirdata", "datasets", "indexes", "fmakv2_index_1.0_sample.json")
)
os.makedirs(os.path.dirname(FMAKV2_INDEX_PATH), exist_ok=True)


def track_to_dict(t, dataset_root):
    track_id = "{:06d}".format(int(t["track_id"]))
    audio_path = "/".join(["000", track_id[:3], track_id + ".mp3"])

    return {
        "000": [audio_path, md5(os.path.join(dataset_root, audio_path))]
    }


def make_fmakv2_index(dataset_root):
    metadata_filename = "metadata_fmakv2.csv"
    metadata_path = os.path.join(dataset_root, "metadata", metadata_filename)

    with open(metadata_path, newline="", encoding="utf-8") as f:
        tracks = {
            t["track_id"]: track_to_dict(t, dataset_root)
            for t in csv.DictReader(f)
        }

    index = {
        "version": "1.0",
        "tracks": tracks,
        "metadata": {
            metadata_filename: (
                "/".join(["metadata", metadata_filename]),
                md5(metadata_path),
            )
        },
    }

    with open(FMAKV2_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)

    print("Index written to:", os.path.abspath(FMAKV2_INDEX_PATH))


def main():
    parser = argparse.ArgumentParser(description="Create index for fmakv2 dataset.")
    parser.add_argument("dataset_root", type=str, help="Path to the dataset root.")
    args = parser.parse_args()
    make_fmakv2_index(args.dataset_root)


if __name__ == "__main__":
    main()
