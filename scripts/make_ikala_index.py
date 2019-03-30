import argparse
import csv
import glob
import hashlib
import json
import os
from urllib import request


ID_MAPPING_URL = "http://mac.citi.sinica.edu.tw/ikala/id_mapping.txt"
IKALA_INDEX_PATH = "../mir_dataset_loaders/indexes/ikala_index.json"


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
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_ikala_index(ikala_data_path):
    lyrics_dir = os.path.join(ikala_data_path, "Lyrics")
    lyrics_files = glob.glob(os.path.join(lyrics_dir, "*.lab"))
    track_ids = sorted(
        [os.path.basename(f).split('.')[0] for f in lyrics_files])

    id_map_path = os.path.join(ikala_data_path, "id_mapping.txt")
    if not os.path.exists(id_map_path):
        request.urlretrieve(ID_MAPPING_URL, filename=id_map_path)

    with open(id_map_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        singer_map = {}
        for line in reader:
            if line[0] == 'singer':
                continue
            singer_map[line[1]] = line[0]

    ikala_index = {k: {} for k in track_ids}
    for key in ikala_index.keys():
        songid = key.split('_')[0]
        audio_checksum = md5(os.path.join(
            ikala_data_path, "Wavfile/{}.wav".format(key)))
        pitch_checksum = md5(os.path.join(
            ikala_data_path, "PitchLabel/{}.pv".format(key)))
        lyrics_checksum = md5(os.path.join(
            ikala_data_path, "Lyrics/{}.lab".format(key)))

        ikala_index[key] = {
            'data_files': {
                'audio': {
                    'path': "iKala/Wavfile/{}.wav".format(key),
                    'checksum': audio_checksum
                },
                'pitch': {
                    'path': "iKala/PitchLabel/{}.pv".format(key),
                    'checksum': pitch_checksum
                },
                'lyrics': {
                    'path': "iKala/Lyrics/{}.lab".format(key),
                    'checksum': lyrics_checksum
                }
            },
            'singer_id': singer_map[songid],
            'song_id': key.split('_')[0],
            'section': key.split('_')[1]
        }

    with open(IKALA_INDEX_PATH, 'w') as fhandle:
        json.dump(ikala_index, fhandle, indent=2)


def main(args):
    make_ikala_index(args.ikala_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make IKala index file.")
    PARSER.add_argument("ikala_data_path",
                        type=str,
                        help="Path to IKala data folder.")

    main(PARSER.parse_args())
