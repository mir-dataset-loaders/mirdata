import argparse
import hashlib
import json
import os


TONAS_INDEX_PATH = '../mirdata/datasets/indexes/tonas_index.json'


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_tonas_index(dataset_data_path):

    tonas_index = {'version': '1.0', 'tracks': {}, 'metadata': []}

    for style in os.listdir(os.path.join(dataset_data_path)):
        if '.' not in style:
            for track in os.listdir(os.path.join(dataset_data_path, style)):
                if '.wav' in track:
                    # Declare track attributes
                    index = track.replace('.wav', '')
                    f0_path = index + '.f0.Corrected'
                    notes_path = index +  '.notes.Corrected'

                    tonas_index['tracks'][index] = {
                        "audio": [
                            os.path.join(style, track),
                            md5(os.path.join(dataset_data_path, style, track))
                        ],
                        "f0": [
                            os.path.join(style, f0_path),
                            md5(os.path.join(dataset_data_path, style, f0_path))
                        ],
                        "notes": [
                            os.path.join(style, notes_path),
                            md5(os.path.join(dataset_data_path, style, notes_path))
                        ],
                    }
    tonas_index['metadata'] = [
        'TONAS-Metadata.txt',
        md5(os.path.join(dataset_data_path, 'TONAS-Metadata.txt')),
    ]

    with open(TONAS_INDEX_PATH, 'w') as fhandle:
        json.dump(tonas_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_tonas_index(args.dataset_data_path)
    print("done!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Make TONAS index file.'
    )
    PARSER.add_argument(
        'dataset_data_path',
        type=str,
        help='Path to TONAS data folder.',
    )

    main(PARSER.parse_args())
