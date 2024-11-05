import argparse
import json
import os
import glob
from mirdata.validate import md5

MDB_STEM_SYNTH_PATH = "mirdata/datasets/indexes/mdb_stem_synth_index_{}.json"


def make_mdb_stem_synth_index(mdb_stem_synth_data_path: str, version: str) -> None:
    annotation_dir = "annotation_stems"
    audio_dir = "audio_stems"

    annotation_path = os.path.join(mdb_stem_synth_data_path, annotation_dir)
    annotation_files = glob.glob(os.path.join(annotation_path, "*.RESYN.csv"))

    audio_path = os.path.join(mdb_stem_synth_data_path, audio_dir)

    track_ids = sorted(
        [os.path.basename(f).replace(".RESYN.csv", "") for f in annotation_files]
    )

    # top-key level tracks
    index_tracks = {
        track_id: {
            "audio": (
                f"{audio_dir}/{track_id}.RESYN.wav",
                md5(os.path.join(audio_path, f"{track_id}.RESYN.wav")),
            ),
            "f0": (
                f"{annotation_dir}/{track_id}.RESYN.csv",
                md5(os.path.join(annotation_path, f"{track_id}.RESYN.csv")),
            ),
        }
        for track_id in track_ids
    }

    # top-key level version
    mdb_stem_synth_index = {
        "version": version,
        "tracks": index_tracks,
    }

    with open(MDB_STEM_SYNTH_PATH.format(version), "w") as fhandle:
        json.dump(mdb_stem_synth_index, fhandle, indent=2)


def main(args):
    make_mdb_stem_synth_index(args.mdb_stem_synth_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make MDB-stem-synth index file.")
    PARSER.add_argument(
        "mdb_stem_synth_data_path",
        type=str,
        help="Path to MDB-stem-synth dataset folder.",
    )
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())
