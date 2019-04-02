from collections import namedtuple
import os

from . import MIR_DATASETS_DIR
from . import md5


def get_local_path(data_home, rel_path):
    if data_home is None:
        return os.path.join(MIR_DATASETS_DIR, rel_path)
    else:
        return os.path.join(data_home, rel_path)


def validator(dataset_index, data_home):
    missing_files = {}
    invalid_checksums = {}
    for track_id, track in dataset_index.items():
        missing_files[track_id] = []
        for key in track.keys():
            filepath = track[key][0]
            checksum = track[key][1]
            local_path = get_local_path(filepath, data_home)
            if not os.path.exists(local_path):
                missing_files[track_id].append(local_path)
            elif md5(local_path) != checksum:
                invalid_checksums[track_id].append(local_path)

    for track_id in missing_files.keys():
        if len(missing_files[track_id]) > 0:
            print("Files missing for {}:".format(track_id))
            for fpath in missing_files[track_id]:
                print(fpath)
            print("-" * 20)

    for track_id in invalid_checksums.keys():
        if len(missing_files[track_id]) > 0:
            print("Invalid checksums for {}:".format(track_id))
            for fpath in missing_files[track_id]:
                print(fpath)
            print("-" * 20)

    return missing_files, invalid_checksums


F0Data = namedtuple(
    'F0Data',
    ['times', 'frequencies', 'confidence']
)

LyricsData = namedtuple(
    'LyricsData',
    ['start_time', 'end_time', 'lyric', 'pronounciation']
)
