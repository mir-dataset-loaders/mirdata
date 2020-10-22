import argparse
import hashlib
import json
import os


beatport_key_INDEX_PATH = '../mirdata/indexes/beatport_key_index.json'
BEATLES_ANNOTATION_SCHEMA = ['JAMS']


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


def make_beatport_key_index(data_path):
    meta_dir = os.path.join(data_path, 'meta')
    audio_dir = os.path.join(data_path, 'audio')
    key_dir = os.path.join(data_path, 'keys')
    beatport_key_index = {}
    for track_id, ann_dir in enumerate(sorted(os.listdir(meta_dir))):
        ann_dir_full = os.path.join(meta_dir, ann_dir)
        if '.json' in ann_dir:
            codec = '.mp3'
            audio_path = os.path.join(audio_dir, ann_dir.replace('.json', codec))
            chord_path = os.path.join(key_dir, ann_dir.replace('.json', '.txt'))
            if "*" in audio_path:
                meta = (None, None)
            else:
                meta = (ann_dir_full.replace(data_path + '/', ''), md5(ann_dir_full))

            beatport_key_index[track_id] = {
                'audio': (audio_path.replace(data_path + '/', ''), md5(audio_path)),
                'meta': meta,
                'key': (chord_path.replace(data_path + '/', ''), md5(chord_path)),
            }
    with open(beatport_key_INDEX_PATH, 'w') as fhandle:
        json.dump(beatport_key_index, fhandle, indent=2)


def main(args):
    make_beatport_key_index(args.beatport_key_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make beatport_key index file.')
    PARSER.add_argument('beatport_key_data_path', type=str, help='Path to beatport_key data folder.')
    main(PARSER.parse_args())
