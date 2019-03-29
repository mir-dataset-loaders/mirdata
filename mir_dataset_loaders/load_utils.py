from collections import namedtuple
import os

from . import MIR_DATASETS_DIR


def get_local_path(data_home, rel_path):
    if data_home is None:
        return os.path.join(MIR_DATASETS_DIR, rel_path)
    else:
        return os.path.join(data_home, rel_path)


def validator(dataset_index, file_keys, data_home):
    missing_files = {}
    for track_id, track in dataset_index.items():
        missing_files[track_id] = []
        for key in file_keys:
            local_path = get_local_path(track[key], data_home)
            if not os.path.exists(local_path):
                missing_files[track_id].append(local_path)

    for track_id in missing_files.keys():
        if len(missing_files[track_id]) > 0:
            print("Files missing for {}:".format(track_id))
            for fpath in missing_files[track_id]:
                print(fpath)
            print("-" * 20)
    return missing_files


F0Data = namedtuple(
    'F0Data',
    ['times', 'frequencies', 'confidence']
)

LyricsData = namedtuple(
    'LyricsData',
    ['start_time', 'end_time', 'lyric', 'pronounciation']
)
