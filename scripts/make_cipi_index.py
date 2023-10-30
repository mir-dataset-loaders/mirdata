import argparse
import json
import os
import sys
from mirdata.validate import md5


def make_cipi_indexes(args):
    version = args.full_version
    cipi_index = {
        "version": version,
        "tracks": {},
        "metadata": [
            "index.json",
            md5(args.path + "/index.json"),
        ]
    }
    # load json /home/mir_datasets/cipi/index.json
    data = json.load(open(args.path + "/index.json", "r"))
    for k, track in data.items():
        cipi_index["tracks"]["cipi_{}".format(k)] = {"annotations": {}}
        cipi_index["tracks"]["cipi_{}".format(k)]["annotations"]["lh_fingering"] = [
            f"ArGNNThumb-s/lh/{k}.pt",
            md5(args.path + f"/ArGNNThumb-s/lh/{k}.pt"),
        ]
        cipi_index["tracks"]["cipi_{}".format(k)]["annotations"]["rh_fingering"] = [
            f"ArGNNThumb-s/rh/{k}.pt",
            md5(args.path + f"/ArGNNThumb-s/rh/{k}.pt"),
        ]
        cipi_index["tracks"]["cipi_{}".format(k)]["annotations"]["expressiviness"] = [
            f"virtuoso/{k}.pt",
            md5(args.path + f"/virtuoso/{k}.pt"),
        ]
        cipi_index["tracks"]["cipi_{}".format(k)]["annotations"]["notes"] = [
            f"k/{k}.pt",
            md5(args.path + f"/k/{k}.pt"),
        ]
        cipi_index["tracks"]["cipi_{}".format(k)]["annotations"]["notes"] = [
            f"k/{k}.pt",
            md5(args.path + f"/k/{k}.pt"),
        ]


    # save json ../mirdata/datasets/indexes/cipi_index.json
    with open(f"../mirdata/datasets/indexes/cipi_index_{version}.json", "w") as fhandle:
        json.dump(cipi_index, fhandle, indent=4, ensure_ascii=False)


def main(args):
    make_cipi_indexes(args)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make cipi index files.")
    PARSER.add_argument("full_version", default="1.0", type=str, help="full index version")
    PARSER.add_argument("path", type=str, help="full index version")
    args = PARSER.parse_args()
    main(args)