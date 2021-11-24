import argparse
import glob
import json
import os
from mirdata.validate import md5

DATASET_ORCHIDEASOL_PATH = "./mirdata/datasets/indexes/orchidea_index_1.0.json"


def make_orchideasol_index(dataset_data_path):
    #annotation_dir = os.path.join(dataset_data_path, "annotation")
    #annotation_files = glob.glob(os.path.join(annotation_dir, "*.lab"))
    #track_ids = sorted([os.path.basename(f).split(".")[0] for f in annotation_files])

    dataset_index = {"version": "1.0", "tracks": {}, "metadata": {}}

    for subfolder in os.listdir(dataset_data_path):
        if "OrchideaSOL2020" == subfolder:
            for classfolder in os.listdir(os.path.join(dataset_data_path, subfolder)):
                if classfolder[0] is not '.':
                    for instrumentfolder in os.listdir(os.path.join(dataset_data_path, subfolder, classfolder)):
                        if instrumentfolder[0] is not '.':
                            for techniquefolder in os.listdir(os.path.join(dataset_data_path, subfolder, classfolder, instrumentfolder)):
                                if techniquefolder[0] is not '.':
                                    for track in os.listdir(os.path.join(dataset_data_path, subfolder, classfolder, instrumentfolder, techniquefolder)):
                                        print(track)
                                        dataset_index["tracks"][track.replace(".wav", "")] = {
                                            "audio": [
                                                os.path.join(subfolder, classfolder, instrumentfolder, techniquefolder, track),
                                                md5(os.path.join(dataset_data_path, subfolder, classfolder, instrumentfolder, techniquefolder, track)),
                                            ]
                                        }

    # top-key level metadata
    metadata_checksums = md5(os.path.join(dataset_data_path, "OrchideaSOL2020.md5.txt"))

    with open(DATASET_ORCHIDEASOL_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    make_orchideasol_index(args.dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make OrchidaSOL index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to OrchideaSOL data folder."
    )

    main(PARSER.parse_args())