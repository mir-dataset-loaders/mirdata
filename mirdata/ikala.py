# -*- coding: utf-8 -*-
"""iKala Dataset Loader

The iKala dataset is comprised of 252 30-second excerpts sampled from 206 iKala
songs (plus 100 hidden excerpts reserved for MIREX).
The music accompaniment and the singing voice are recorded at the left and right
channels respectively and can be found under the Wavfile directory.
In addition, the human-labeled pitch contours and timestamped lyrics can be
found under PitchLabel and Lyrics respectively.

Details can be found at http://mac.citi.sinica.edu.tw/ikala/


Attributes:
    DATASET_DIR (str): The directory name for iKala dataset. Set to `'iKala'`.

    DATA.index (dict): {track_id: track_data}.
        track_data is a `IKalaTrack` namedtuple.

    TIME_STEP (float): Time step unit (in second) (TODO: what is this? hop length? window?)

    DATA.metadata (None): TODO

    ID_MAPPING_URL (str): URL to get id-to-url mapping text file

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import librosa
import logging
import numpy as np

import mirdata.utils as utils
import mirdata.download_utils as download_utils

DATASET_DIR = 'iKala'
TIME_STEP = 0.032  # seconds
ID_MAPPING_REMOTE = download_utils.RemoteFileMetadata(
    filename='id_mapping.txt',
    url='http://mac.citi.sinica.edu.tw/ikala/id_mapping.txt',
    checksum='81097b587804ce93e56c7a331ba06abc',
    destination_dir=None,
)


def _load_metadata(data_home):
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    id_map_path = os.path.join(data_home, 'id_mapping.txt')
    if not os.path.exists(id_map_path):
        logging.info(
            'Metadata file {} not found.'.format(id_map_path)
            + 'You can download the metadata file for ikala by running ikala.download'
        )
        return None

    with open(id_map_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter='\t')
        singer_map = {}
        for line in reader:
            if line[0] == 'singer':
                continue
            singer_map[line[1]] = line[0]

    singer_map['data_home'] = data_home

    return singer_map


DATA = utils.LargeData('ikala_index.json', _load_metadata)


class Track(object):
    """iKala track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        track_id (str): track id
        audio_path (str): track audio path
        song_id (str): song id of the track
        section (str): section (todo)
        singer_id (str): singer id
        f0 (F0Data): pitch
        lyrics (LyricData): lyrics

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in iKala'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        metadata = DATA.metadata(data_home)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.song_id = track_id.split('_')[0]
        self.section = track_id.split('_')[1]

        if metadata is not None and self.song_id in metadata:
            self.singer_id = metadata[self.song_id]
        else:
            self.singer_id = None

    def __repr__(self):
        repr_string = (
            "iKala Track(track_id={}, audio_path={}, song_id={}, "
            + "section={}, singer_id={}, "
            + "f0=F0Data('times', 'frequencies', 'confidence'), "
            + "lyrics=LyricData('start_times', 'end_times', 'lyrics', 'pronounciations'))"
        )
        return repr_string.format(
            self.track_id, self.audio_path, self.song_id, self.section, self.singer_id
        )

    @utils.cached_property
    def f0(self):
        return _load_f0(os.path.join(self._data_home, self._track_paths['pitch'][0]))

    @utils.cached_property
    def lyrics(self):
        return _load_lyrics(
            os.path.join(self._data_home, self._track_paths['lyrics'][0])
        )

    @property
    def vocal_audio(self):
        """Load iKala vocal audio

        Returns:
            vocal_channel (np.array): vocal audio. size of `(N, )`
            sr (int): sampling rate of the audio file
        """
        audio, sr = librosa.load(self.audio_path, sr=None, mono=False)
        vocal_channel = audio[1, :]
        return vocal_channel, sr

    @property
    def instrumental_audio(self):
        """Load iKala instrumental audio

        Returns:
            instrumental_channel (np.array): vocal audio. size of `(N, )`
            sr (int): sampling rate of the audio file
        """
        audio, sr = librosa.load(self.audio_path, sr=None, mono=False)
        instrumental_channel = audio[0, :]
        return instrumental_channel, sr

    @property
    def mix_audio(self):
        """Load iKala mixture audio

        Returns:
            mixed_audio (np.array): vocal audio. size of `(2, N)`
            sr (int): sampling rate of the audio file
        """
        mixed_audio, sr = librosa.load(self.audio_path, sr=None, mono=True)
        # multipy by 2 because librosa averages the left and right channel.
        return 2.0 * mixed_audio, sr


def download(data_home=None, force_overwrite=False):
    """Download iKala Dataset. However, iKala dataset is not available for
    download anymore. This function prints a helper message to organize
    pre-downloaded iKala dataset.

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool): If True, existing files are overwritten by the
            downloaded files.
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_message = """
        Unfortunately the iKala dataset is not available for download.
        If you have the iKala dataset, place the contents into a folder called
        {ikala_dir} with the following structure:
            > {ikala_dir}/
                > Lyrics/
                > PitchLabel/
                > Wavfile/
        and copy the {ikala_dir} folder to {save_path}
    """.format(
        ikala_dir=DATASET_DIR, save_path=data_home
    )

    download_utils.downloader(
        data_home,
        file_downloads=[ID_MAPPING_REMOTE],
        info_message=download_message,
        force_overwrite=force_overwrite,
    )


def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


def load(data_home=None):
    """Load iKala dataset

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """

    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    ikala_data = {}
    for key in track_ids():
        ikala_data[key] = Track(key, data_home=data_home)
    return ikala_data


def _load_f0(f0_path):
    if not os.path.exists(f0_path):
        return None

    with open(f0_path) as fhandle:
        lines = fhandle.readlines()
    f0_midi = np.array([float(line) for line in lines])
    f0_hz = librosa.midi_to_hz(f0_midi) * (f0_midi > 0)
    confidence = (f0_hz > 0).astype(float)
    times = (np.arange(len(f0_midi)) * TIME_STEP) + (TIME_STEP / 2.0)
    f0_data = utils.F0Data(times, f0_hz, confidence)
    return f0_data


def _load_lyrics(lyrics_path):
    if not os.path.exists(lyrics_path):
        return None
    # input: start time (ms), end time (ms), lyric, [pronunciation]
    with open(lyrics_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=' ')
        start_times = []
        end_times = []
        lyrics = []
        pronunciations = []
        for line in reader:
            start_times.append(float(line[0]) / 1000.0)
            end_times.append(float(line[1]) / 1000.0)
            lyrics.append(line[2])
            if len(line) > 2:
                pronunciation = ' '.join(line[3:])
                pronunciations.append(pronunciation if pronunciation != '' else None)
            else:
                pronunciations.append(None)

    lyrics_data = utils.LyricData(
        np.array(start_times),
        np.array(end_times),
        np.array(lyrics),
        np.array(pronunciations),
    )
    return lyrics_data


def cite():
    """Print the reference"""
    cite_data = """
=========== MLA ===========
Chan, Tak-Shing, et al.
"Vocal activity informed singing voice separation with the iKala dataset."
2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

========== Bibtex ==========
@inproceedings{chan2015vocal,
    title={Vocal activity informed singing voice separation with the iKala dataset},
    author={Chan, Tak-Shing and Yeh, Tzu-Chun and Fan, Zhe-Cheng and Chen, Hung-Wei and Su, Li and Yang, Yi-Hsuan and Jang, Roger},
    booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={718--722},
    year={2015},
    organization={IEEE}
}
"""
    print(cite_data)
