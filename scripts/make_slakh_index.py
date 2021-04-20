import argparse
import glob
import json
import os

import yaml

from mirdata.validate import md5

DATASET_INDEX_PATH = "mirdata/datasets/indexes/slakh_index_{}.json"


def get_file_info(path):
    if os.path.exists(path):
        return [path, md5(path)]
    else:
        print("warning: {} not found".format(path))
        return [None, None]


def make_dataset_index(dataset_data_path, version):
    curr_dir = os.getcwd()
    os.chdir(dataset_data_path)

    dataset_index_path = DATASET_INDEX_PATH.format(version)

    if version == "baby":
        splits = [""]
        topdir = "babyslakh_16k"
        fmt = "wav"
    else:
        splits = ["train", "validation", "test"]
        topdir = "slakh2100_flac_redux"
        fmt = "flac"

    multitrack_index = {}
    track_index = {}

    for split in splits:
        mtrack_ids = sorted(
            [
                os.path.basename(folder)
                for folder in glob.glob(os.path.join(topdir, split, "Track*"))
            ]
        )
        for mtrack_id in mtrack_ids:
            mtrack_path = os.path.join(topdir, split, mtrack_id)
            metadata_path = os.path.join(mtrack_path, "metadata.yaml")
            with open(metadata_path, "r") as fhandle:
                metadata = yaml.safe_load(fhandle)

            track_ids = metadata["stems"].keys()
            midi_path = os.path.join(mtrack_path, "all_src.mid")
            mix_path = os.path.join(mtrack_path, "mix.{}".format(fmt))

            multitrack_index[mtrack_id] = {
                "tracks": [
                    "{}-{}".format(mtrack_id, track_id) for track_id in track_ids
                ],
                "midi": get_file_info(midi_path),
                "mix": get_file_info(mix_path),
                "metadata": get_file_info(metadata_path),
            }

            for track_id in track_ids:
                # TODO - check what to do if neither the midi nor audio exists
                # but the stem is part of the metadata
                audio_path = os.path.join(
                    mtrack_path, "stems", "{}.{}".format(track_id, fmt)
                )
                midi_path = os.path.join(mtrack_path, "MIDI", "{}.mid".format(track_id))
                track_index["{}-{}".format(mtrack_id, track_id)] = {
                    "audio": get_file_info(audio_path),
                    "midi": get_file_info(midi_path),
                    "metadata": get_file_info(metadata_path),
                }

    # top-key level version
    dataset_index = {
        "version": version,
        "tracks": track_index,
        "multitracks": multitrack_index,
    }

    os.chdir(curr_dir)
    with open(dataset_index_path, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    make_dataset_index(args.dataset_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to dataset data folder."
    )
    PARSER.add_argument("version", type=str, help="Dataset version.")
    main(PARSER.parse_args())
