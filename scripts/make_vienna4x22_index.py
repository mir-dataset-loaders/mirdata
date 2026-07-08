"""Generate the mirdata index for the Vienna 4x22 Piano Corpus.

Point ``data_path`` at a local checkout of https://github.com/CPJKU/vienna4x22
(or an extracted copy of the pinned tarball) that contains ``musicxml/``,
``midi/`` and ``match/`` sub-directories.

Usage:
    python make_vienna4x22_index.py /path/to/vienna4x22
"""

import argparse
import hashlib
import json
import os

INDEX_PATH = "../mirdata/datasets/indexes/vienna4x22_index_1.0.json"

PIECES = (
    "Chopin_op10_no3",
    "Chopin_op38",
    "Mozart_K331_1st-mov",
    "Schubert_D783_no15",
)
PIANIST_IDS = tuple(f"{i:02d}" for i in range(1, 23))


def md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def make_vienna4x22_index(data_path, index_path=INDEX_PATH):
    tracks = {}
    for piece in PIECES:
        score_rel = os.path.join("musicxml", f"{piece}.musicxml")
        score_abs = os.path.join(data_path, score_rel)
        score_md5 = md5(score_abs)
        for pid in PIANIST_IDS:
            track_id = f"{piece}_p{pid}"
            perf_rel = os.path.join("midi", f"{track_id}.mid")
            match_rel = os.path.join("match", f"{track_id}.match")
            tracks[track_id] = {
                "score": (score_rel, score_md5),
                "performance": (perf_rel, md5(os.path.join(data_path, perf_rel))),
                "match": (match_rel, md5(os.path.join(data_path, match_rel))),
            }

    index = {"version": "1.0", "tracks": tracks}
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def main(args):
    make_vienna4x22_index(args.data_path, args.index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Vienna 4x22 index.")
    parser.add_argument("data_path", type=str, help="Path to vienna4x22 folder.")
    parser.add_argument("--index_path", type=str, default=INDEX_PATH)
    main(parser.parse_args())
