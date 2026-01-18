import argparse
import os
import glob
from mirdata.validate import md5
import json


dataset_data_path = r"C:\Users\wispi\Desktop\EGSet12"
EGSET12_INDEX_PATH = "mirdata/datasets/indexes/egset12_index_{}.json"


def make_egset12_index(dataset_data_path, version):
    annotation_dir = dataset_data_path
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.jams"))
    track_ids = sorted([os.path.basename(f).split(".")[0] for f in annotation_files])
    # top-key level metadata
    index_metadata = {"metadata": {}}
    # top-key level tracks
    index_tracks = {}
    for track_id in track_ids:
        audio_checksum = md5(os.path.join(dataset_data_path, "{}.wav".format(track_id)))
        annotation_checksum = md5(
            os.path.join(dataset_data_path, "{}.jams".format(track_id))
        )
        index_tracks[track_id] = {
            "audio": ("{}.wav".format(track_id), audio_checksum),
            "annotation": ("{}.jams".format(track_id), annotation_checksum),
        }
    # top-key level version
    dataset_index = {"version": version}
    # combine all in dataset index
    dataset_index.update(index_metadata)
    dataset_index.update({"tracks": index_tracks})
    with open(EGSET12_INDEX_PATH.format(version), "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)
    print(f"Index file created:{EGSET12_INDEX_PATH.format(version)}")


def main(args):
    make_egset12_index(args.dataset_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="MakeEGSet12 index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to EGSet12 data folder."
    )
    PARSER.add_argument("version", type=str, help="Dataset version (e.g., '1.0')")
    main(PARSER.parse_args())
