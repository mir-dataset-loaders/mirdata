import argparse
import hashlib
import json
import glob
import os
from mirdata.validate import md5


OTMM_MAKAM_INDEX_PATH = "/Users/genisplaja/Desktop/mirdata/mirdata/datasets/indexes/compmusic_indian_tonic.json"


def make_compmusic_indian_tonic(dataset_data_path):

    tonic_index = {"version": "1.0", "tracks": {}, "metadata": {}}

    for center_fold in glob.glob(os.path.join(dataset_data_path, "*/")):
        center = center_fold.split("/")[-2]
        for metafile in glob.glob(os.path.join(center_fold, "annotations", center + "*.json")):
            if "IITM1" not in metafile:
                with open(metafile) as fhandle:
                    meta = json.load(fhandle)
                    files = list(meta.keys())
                    for fil in files:
                        idx = fil.split("/")[-1].replace(".mp3", "")
                        tonic_index["tracks"][idx] = {
                            "audio": [
                                fil,
                                md5(os.path.join(dataset_data_path.replace("/TonicDataset", ""), fil)),
                            ]
                        }

    tonic_index["metadata"]["CM1"] = [
        os.path.join("TonicDataset", "CM", "annotations", "CM1.json"),
        md5(os.path.join(dataset_data_path, "CM", "annotations", "CM1.json")),
    ]
    tonic_index["metadata"]["CM2"] = [
        os.path.join("TonicDataset", "CM", "annotations", "CM2.json"),
        md5(os.path.join(dataset_data_path, "CM", "annotations", "CM2.json")),
    ]
    tonic_index["metadata"]["CM3"] = [
        os.path.join("TonicDataset", "CM", "annotations", "CM3.json"),
        md5(os.path.join(dataset_data_path, "CM", "annotations", "CM3.json")),
    ]
    tonic_index["metadata"]["IISc"] = [
        os.path.join("TonicDataset", "IISc", "annotations", "IISc.json"),
        md5(os.path.join(dataset_data_path, "IISc", "annotations", "IISc.json")),
    ]
    tonic_index["metadata"]["IITM2"] = [
        os.path.join("TonicDataset", "IITM", "annotations", "IITM2.json"),
        md5(os.path.join(dataset_data_path, "IITM", "annotations", "IITM2.json")),
    ]

    with open(OTMM_MAKAM_INDEX_PATH, "w") as fhandle:
        json.dump(tonic_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_compmusic_indian_tonic(args.dataset_data_path)
    print("done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make CompMusic Tonic Dataset index file."
    )
    PARSER.add_argument(
        "dataset_data_path",
        type=str,
        help="Path to ompMusic Tonic Dataset data folder.",
    )

    main(PARSER.parse_args())
