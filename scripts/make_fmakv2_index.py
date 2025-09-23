import argparse
import csv
import json
import os
from mirdata.validate import md5

FMAKV2_INDEX_PATH = "/mirdata/datasets/indexes/fmakv2_index_1.0.json"

def track_to_dict(t, data_path, id_col):
    track_id = "{:06d}".format(int(t[id_col]))
    audio_path = track_id[:3] + "/" + track_id + ".mp3"
    return {
        "audio": [audio_path, md5(os.path.join(data_path, audio_path))],
    }

def make_fmakv2_index(data_path, metadata_filename, id_col):
    metadata_file = os.path.join(data_path, metadata_filename)

    with open(metadata_file, newline="", encoding="utf-8") as f:
        tracks = {
            t[id_col]: track_to_dict(t, data_path, id_col)
            for t in csv.DictReader(f)
        }

    fmakv2_index = {
        "version": "1.0",
        "tracks": tracks,
        "metadata": {metadata_filename: (metadata_filename, md5(metadata_file))},
    }

    os.makedirs(os.path.dirname(FMAKV2_INDEX_PATH), exist_ok=True)
    with open(FMAKV2_INDEX_PATH, "w", encoding="utf-8") as fhandle:
        json.dump(fmakv2_index, fhandle, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Make fmakv2 index file.")
    parser.add_argument("fmakv2_data_path", type=str, help="Path to fmakv2 data folder.")
    parser.add_argument("--metadata-filename", type=str, default="fmakv2_metadata.csv",
                        help="CSV filename (default: fmakv2_metadata.csv)")
    parser.add_argument("--id-col", type=str, default="track_id",
                        help="Column name for IDs (default: track_id)")
    args = parser.parse_args()
    make_fmakv2_index(args.fmakv2_data_path, args.metadata_filename, args.id_col)

if __name__ == "__main__":
    main()
