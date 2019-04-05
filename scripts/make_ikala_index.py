import argparse
import glob
import hashlib
import json
import os


IKALA_INDEX_PATH = "../mirdata/indexes/ikala_index.json"


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

    ikala_index = {}
    for track_id in track_ids:
        audio_checksum = md5(os.path.join(
            ikala_data_path, "Wavfile/{}.wav".format(track_id)))
        pitch_checksum = md5(os.path.join(
            ikala_data_path, "PitchLabel/{}.pv".format(track_id)))
        lyrics_checksum = md5(os.path.join(
            ikala_data_path, "Lyrics/{}.lab".format(track_id)))

        ikala_index[track_id] = {
            'audio': (
                "iKala/Wavfile/{}.wav".format(track_id),
                audio_checksum
            ),
            'pitch': (
                "iKala/PitchLabel/{}.pv".format(track_id),
                pitch_checksum
            ),
            'lyrics': (
                "iKala/Lyrics/{}.lab".format(track_id),
                lyrics_checksum
            )
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
