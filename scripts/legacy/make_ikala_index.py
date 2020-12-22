# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
from mirdata.utils import md5

IKALA_INDEX_PATH = "../mirdata/datasets/indexes/ikala_index.json"


def make_ikala_index(ikala_data_path):
    lyrics_dir = os.path.join(ikala_data_path, 'Lyrics')
    lyrics_files = glob.glob(os.path.join(lyrics_dir, '*.lab'))
    track_ids = sorted([os.path.basename(f).split('.')[0] for f in lyrics_files])

    # top-key level metadata
    metadata_checksum = md5(os.path.join(ikala_data_path, 'id_mapping.txt'))
    index_metadata = {"metadata":{"id_mapping": ("id_mapping.txt", metadata_checksum)}}

    # top-key level tracks
    index_tracks = {}
    for track_id in track_ids:
        audio_checksum = md5(
            os.path.join(ikala_data_path, "Wavfile/{}.wav".format(track_id))
        )
        pitch_checksum = md5(
            os.path.join(ikala_data_path, "PitchLabel/{}.pv".format(track_id))
        )
        lyrics_checksum = md5(
            os.path.join(ikala_data_path, "Lyrics/{}.lab".format(track_id))
        )

        index_tracks[track_id] = {
            "audio": ("Wavfile/{}.wav".format(track_id), audio_checksum),
            "pitch": ("PitchLabel/{}.pv".format(track_id), pitch_checksum),
            "lyrics": ("Lyrics/{}.lab".format(track_id), lyrics_checksum),
        }

    # top-key level version
    ikala_index = {"version": None}

    # combine all in dataset index
    ikala_index.update(index_metadata)
    ikala_index.update({"tracks": index_tracks})

    with open(IKALA_INDEX_PATH, 'w') as fhandle:
        json.dump(ikala_index, fhandle, indent=2)


def main(args):
    make_ikala_index(args.ikala_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make IKala index file.')
    PARSER.add_argument('ikala_data_path', type=str, help='Path to IKala data folder.')

    main(PARSER.parse_args())
