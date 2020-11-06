import argparse
import hashlib
import json
import os


classicalDB_INDEX_PATH = '../mirdata/indexes/classicalDB_index.json'
CLASSICALDB_ANNOTATION_SCHEMA = ['JAMS']


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


def make_classicalDB_index(data_path):
    audio_dir = os.path.join(data_path, 'audio')
    key_dir = os.path.join(data_path, 'keys')
    classicalDB_index = {}
    for track_id, key_dir in enumerate(sorted(os.listdir(key_dir))):
        if '.txt' in key_dir:
            codec = '.mp3'
            audio_path = os.path.join(audio_dir, key_dir.replace('.json', codec))
            key_path = os.path.join(key_dir, key_dir.replace('.json', '.txt'))

            classicalDB_index[track_id] = {
                'audio': (audio_path.replace(data_path + '/', ''), md5(audio_path)),
                'key': (key_path.replace(data_path + '/', ''), md5(key_path)),
            }
    with open(classicalDB_INDEX_PATH, 'w') as fhandle:
        json.dump(classicalDB_index, fhandle, indent=2)


def main(args):
    make_classicalDB_index(args.classicalDB_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make classicalDB index file.')
    PARSER.add_argument('classicalDB_data_path', type=str, help='Path to classicalDB data folder.')
    main(PARSER.parse_args())
