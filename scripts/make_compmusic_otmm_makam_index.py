import argparse
import hashlib
import json
import os
from mirdata.validate import md5


OTMM_MAKAM_INDEX_PATH = '../mirdata/datasets/indexes/otmm_makam_index.json'


def make_otmm_makam_index(dataset_data_path):

    otmm_index = {'version': 'dlfm2016', 'tracks': {}, 'metadata': []}

    for makam in os.listdir(os.path.join(dataset_data_path, 'data')):
        if '.' not in makam:
            for track in os.listdir(os.path.join(dataset_data_path, 'data', makam)):
                if '.json' in track:
                    # Declare track attributes
                    index = track.split('.json')[0]
                    pitch_path = index + '.pitch'

                    otmm_index['tracks'][index] = {
                        "metadata": [
                            os.path.join(
                                'MTG-otmm_makam_recognition_dataset-f14c0d0',
                                'data',
                                makam,
                                track,
                            ),
                            md5(os.path.join(dataset_data_path, 'data', makam, track)),
                        ],
                        "pitch": [
                            os.path.join(
                                'MTG-otmm_makam_recognition_dataset-f14c0d0',
                                'data',
                                makam,
                                pitch_path,
                            ),
                            md5(
                                os.path.join(
                                    dataset_data_path, 'data', makam, pitch_path
                                )
                            ),
                        ],
                    }
    otmm_index['metadata'] = [
        os.path.join('MTG-otmm_makam_recognition_dataset-f14c0d0', 'annotations.json'),
        md5(os.path.join(dataset_data_path, 'annotations.json')),
    ]

    with open(OTMM_MAKAM_INDEX_PATH, 'w') as fhandle:
        json.dump(otmm_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_otmm_makam_index(args.dataset_data_path)
    print("done!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Make OTMM Makam Recognition index file.'
    )
    PARSER.add_argument(
        'dataset_data_path',
        type=str,
        help='Path to OTMM Makam Recognition data folder.',
    )

    main(PARSER.parse_args())
