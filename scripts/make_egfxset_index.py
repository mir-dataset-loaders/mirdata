import argparse
import csv
import json
import os
from mirdata.validate import md5

EGFXSET_INDEX_PATH = "mirdata/datasets/indexes/egfxsetset_index_{}.json"


def make_egfxset_index(egfxset_data_path: str, version: str) -> None:
    assert version == "1"
    annotations_dir = os.path.join(egfxset_data_path, "Annotations")

    audio_dir = os.path.join(egfxset_data_path, "Audio")

    metadata_path = os.path.join(egfxset_data_path, "egfxset_metadata.csv")
    with open(metadata_path) as f:
        track_ids = sorted([row["track_id"] for row in csv.DictReader(f)])

    # top-key level tracks
    index_tracks = {
        track_id: {
            "audio": (
                f"Audio/egfxset_{track_id}.wav",
                md5(os.path.join(audio_dir, f"egfxset_{track_id}.wav")),
            ),
        }
        for track_id in track_ids
    }

    # top-key level version
    egfxset_index = {
        "version": version,
        "tracks": index_tracks,
        "metadata": {
            "egfxset_metadata": ("egfxset_metadata.csv", md5(metadata_path))
        },
    }

    with open(EGFXSET_INDEX_PATH.format(version), "w") as fhandle:
        json.dump(egfxset_index, fhandle, indent=2)


def main(args):
    make_egfxset_index(args.egfxset_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make EGFxSet index file.")
    PARSER.add_argument(
        "egfxset_data_path", type=str, help="Path to EGFxSet data folder."
    )
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())
