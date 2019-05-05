import argparse
import hashlib
import json
import os


RWC_JAZZ_INDEX_PATH = "../mirdata/indexes/rwc_jazz_index.json"


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


def make_rwc_jazz_index(data_path):
    annotations_dir = os.path.join(data_path, 'RWC-Jazz', 'annotations')
    audio_dir = os.path.join(data_path, 'RWC-Jazz', 'audio')
    annotations_files = os.listdir(os.path.join(annotations_dir,
                                                'AIST.RWC-MDB-J-2001.CHORUS'))
    track_ids = sorted(
        [os.path.basename(f).split('.')[0] for f in annotations_files
         if not f == 'README.TXT'])

    rwc_jazz_index = {}
    for track_id in track_ids:
        # audio
        audio_checksum = None #md5(os.path.join(audio_dir, "{}.wav".format(track_id)))
        annot_checksum, annot_rels = [], []

        # using existing annotations (version 2.0)
        for f in ['CHORUS', 'BEAT']:
            if os.path.exists(os.path.join(annotations_dir,
                                           'AIST.RWC-MDB-J-2001.{}'.format(f), '{}.{}.TXT'.format(track_id, f))):
                annot_checksum.append(md5(os.path.join(annotations_dir,
                                           'AIST.RWC-MDB-J-2001.{}'.format(f), '{}.{}.TXT'.format(track_id, f))))
                annot_rels.append(os.path.join('RWC-Jazz', 'annotations',
                                           'AIST.RWC-MDB-J-2001.{}'.format(f), '{}.{}.TXT'.format(track_id, f)))
            else:
                annot_checksum.append(None)
                annot_rels.append(None)

        rwc_jazz_index[track_id] = {
            'audio': (
                None, #os.path.join('RWC-Jazz', 'audio', "{}.wav".format(track_id)),
                audio_checksum
            ),
            'sections': (
                annot_rels[0],
                annot_checksum[0]
            ),
            'beats': (
                annot_rels[1],
                annot_checksum[1]
            )
        }

    with open(RWC_JAZZ_INDEX_PATH, 'w') as fhandle:
        json.dump(rwc_jazz_index, fhandle, indent=2)


def main(args):
    make_rwc_jazz_index(args.rwc_jazz_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make RWC-Jazz index file.")
    PARSER.add_argument("rwc_jazz_data_path",
                        type=str,
                        help="Path to RWC-Jazz data folder.")

    main(PARSER.parse_args())

