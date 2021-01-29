import os
import sys
import json
import hashlib
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GTZAN_GENRE_INDEX_PATH = os.path.join(
    SCRIPT_DIR, "../mirdata/indexes/gtzan_genre_index.json"
)

sys.path.append(os.path.join(SCRIPT_DIR, "mirdata"))
from mirdata.utils import md5


def make_gtzan_genre_index(data_path):
    index = {}
    for track_key, path in iter_paths(data_path):
        abspath = os.path.join(data_path, path)
        if not os.path.exists(abspath):
            print("Missing file: {}".format(abspath))
            continue

        checksum = md5(abspath)
        audio_path = os.path.join("gtzan_genre/genres", path)
        index[track_key] = {"audio": [audio_path, checksum]}

    with open(GTZAN_GENRE_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def iter_paths(data_path):
    with open(os.path.join(data_path, "bextract_single.mf")) as f:
        for line in f:
            if not line.strip():  # blank space
                continue

            au_path, _ = line.split("\t")
            _, folder, au_filename = au_path.rsplit("/", 2)
            track_key, _ = os.path.splitext(au_filename)
            wav_filename = track_key + ".wav"
            path = os.path.join(folder, wav_filename)
            yield track_key, path


def main():
    parser = argparse.ArgumentParser(description="Make GTZAN-Genre index file.")
    parser.add_argument(
        "gtzan_genre_data_path", type=str, help="Path to the GTZAN-Genre data folder."
    )
    args = parser.parse_args()
    make_gtzan_genre_index(os.path.expanduser(args.gtzan_genre_data_path))


if __name__ == "__main__":
    main()
