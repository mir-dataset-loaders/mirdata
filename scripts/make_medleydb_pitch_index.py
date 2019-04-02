import argparse
import hashlib
import json
import os


MEDLEYDB_PITCH_INDEX_PATH = \
    "../mir_dataset_loaders/indexes/medleydb_pitch_index.json"


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


def make_medleydb_pitch_index(data_path):
    metadata_path = os.path.join(
        data_path, "MedleyDB-Pitch", "medleydb_pitch_metadata.json")
    with open(metadata_path, 'r') as fhandle:
        metadata = json.load(fhandle)

    pitch_index = {}
    for trackid in metadata.keys():
        audio_path = os.path.join(data_path, metadata[trackid]['audio_path'])
        audio_checksum = md5(audio_path)
        local_pitch_path = os.path.join(
            data_path, metadata[trackid]['pitch_path']
        )
        pitch_checksum = md5(local_pitch_path)

        fullid = os.path.basename(audio_path).split('.')[0]
        pitch_index[fullid] = {
            'audio': (
                metadata[trackid]['audio_path'],
                audio_checksum
            ),
            'pitch': (
                metadata[trackid]['pitch_path'],
                pitch_checksum
            )
        }

    with open(MEDLEYDB_PITCH_INDEX_PATH, 'w') as fhandle:
        json.dump(pitch_index, fhandle, indent=2)


def main(args):
    make_medleydb_pitch_index(args.mdb_pitch_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make MedleyDB-Pitch index file.")
    PARSER.add_argument("mdb_pitch_data_path",
                        type=str,
                        help="Path to MedleyDB-Pitch data folder.")

    main(PARSER.parse_args())
