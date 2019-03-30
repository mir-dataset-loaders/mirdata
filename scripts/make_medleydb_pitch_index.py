import argparse
import hashlib
import json
import medleydb as mdb
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

    mtracks = mdb.utils.load_all_multitracks(dataset_version=['V1'])
    index = {}
    for mtrack in mtracks:
        if mtrack.has_bleed:
            continue
        for stem in mtrack.stems.values():
            if stem.pitch_annotation is not None:
                if stem.f0_type != ['m']:
                    continue

                if mtrack.track_id not in index.keys():
                    index[mtrack.track_id] = []
                index[mtrack.track_id].append(stem.stem_idx)

    pitch_index = {}
    for trackid in index.keys():
        mtrack = mdb.MultiTrack(trackid)
        for stemid in index[trackid]:
            audio_path = mtrack.stems[stemid].audio_path
            audio_checksum = md5(audio_path)
            local_pitch_path = os.path.join(
                data_path, 'pitch',
                os.path.basename(mtrack.stems[stemid].pitch_path)
            )
            pitch_checksum = md5(local_pitch_path)

            fullid = os.path.basename(audio_path).split('.')[0]
            pitch_index[fullid] = {
                'data_files': {
                    'audio': {
                        'path': os.path.join(
                            'MedleyDB-Pitch', 'audio',
                            os.path.basename(audio_path)),
                        'checksum': audio_checksum
                    },
                    'pitch': {
                        'path': os.path.join(
                            'MedleyDB-Pitch', 'pitch',
                            os.path.basename(mtrack.stems[stemid].pitch_path)),
                        'checksum': pitch_checksum
                    }
                },
                'instrument': mtrack.stems[stemid].instrument[0],
                'artist': mtrack.artist,
                'title': mtrack.title,
                'genre': mtrack.genre,
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
