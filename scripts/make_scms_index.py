import argparse
import hashlib
import glob
import json
import os
from mirdata.validate import md5


SCMS_INDEX_PATH = '/home/genis/mirdata/mirdata/datasets/indexes/scms_index.json'

def make_scms_index(dataset_data_path):

    scms_index = {
        'version': 1.0,
        'tracks': {},
    }
    dataset_folder_name = 'SCMS'
    for rec in glob.glob(os.path.join(dataset_data_path, 'audio', '*.wav')):

        audio_filename = rec.split('/')[-1]
        idx = audio_filename.replace('.wav', '')
        
        scms_index['tracks'][idx] = {
            'audio': (
                os.path.join(dataset_folder_name, 'audio', rec.split('/')[-1]), 
                md5(os.path.join(dataset_data_path, 'audio', audio_filename))
            ),
            'melody': (
                os.path.join(dataset_folder_name, 'annotations', 'melody', rec.split('/')[-1].replace('.wav', '.csv')),
                md5(os.path.join(dataset_data_path, 'annotations', 'melody', audio_filename.replace('.wav', '.csv')))
            ),
            'activations': (
                os.path.join(dataset_folder_name, 'annotations', 'activations', rec.split('/')[-1].replace('.wav', '.lab')),
                md5(os.path.join(dataset_data_path, 'annotations', 'activations', audio_filename.replace('.wav', '.lab')))
            )
        }

    scms_index["metadata"] = {
        "artists-to-track-mapping": (
            os.path.join(dataset_folder_name, "artists_to_track_mapping.json"),
            md5(os.path.join(dataset_data_path, "artists_to_track_mapping.json")),
        ),
        "train-test-metadata": (
            os.path.join(dataset_folder_name, "metadata.json"),
            md5(os.path.join(dataset_data_path, "metadata.json")),
        )
    }

    with open(SCMS_INDEX_PATH, 'w') as fhandle:
        json.dump(scms_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_scms_index(args.dataset_data_path)
    print("done!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make Saraga-Carnatic-Melody-Synth index file.')
    PARSER.add_argument(
        'dataset_data_path', type=str, help='Path to Saraga-Carnatic-Melody-Synth data folder.'
    )

    main(PARSER.parse_args())
