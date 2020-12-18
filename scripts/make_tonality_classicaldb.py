import argparse
import hashlib
import json
import os


classicalDB_INDEX_PATH = '../mirdata/datasets/indexes/tonality_classicaldb_index.json'
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
    spectrum_dir = os.path.join(data_path, 'spectrums')
    HPCP_dir = os.path.join(data_path, 'HPCPs')
    mb_dir = os.path.join(data_path, 'musicbrainz_metadata')

    classicalDB_index = {}
    for track_id, key_file in enumerate(sorted(os.listdir(key_dir))):
        if '.txt' in key_file:
            codec = '.wav'
            audio_path = os.path.join(audio_dir, os.path.splitext(key_file)[0] + codec)
            spectrum_path = os.path.join(spectrum_dir, os.path.splitext(key_file)[0] + '.json')
            HPCP_path = os.path.join(HPCP_dir, os.path.splitext(key_file)[0] + '.json')
            mb_path = os.path.join(mb_dir, os.path.splitext(key_file)[0] + '.json')
            key_path = os.path.join(key_dir, key_file)

            classicalDB_index[track_id] = {
                'audio': (audio_path.replace(data_path + '/', ''), md5(audio_path)),
                'key': (key_path.replace(data_path + '/', ''), md5(key_path)),
                'spectrum': (spectrum_path.replace(data_path + '/', ''), md5(spectrum_path)),
                'mb': (mb_path.replace(data_path + '/', ''), md5(mb_path)),
                'HPCP': (HPCP_path.replace(data_path + '/', ''), md5(HPCP_path))
            }
    with open(classicalDB_INDEX_PATH, 'w') as fhandle:
        json.dump(classicalDB_index, fhandle, indent=2)


def main(args):
    make_classicalDB_index(args.classicalDB_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make classicalDB index file.')
    PARSER.add_argument('classicalDB_data_path', type=str, help='Path to classicalDB data folder.')
    main(PARSER.parse_args())
