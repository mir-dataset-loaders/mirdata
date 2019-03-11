from collections import namedtuple
import os

from . import MIR_DATASETS_DIR


def abs_path(rel_path, data_home):
    if data_home is None:
        return os.path.join(MIR_DATASETS_DIR, rel_path)
    else:
        return os.path.join(data_home, rel_path)


F0Data = namedtuple(
    'F0Data',
    ['times', 'frequencies', 'confidence']
)

LyricsData = namedtuple(
    'LyricsData',
    ['start_time', 'end_time', 'lyric', 'pronounciation']
)