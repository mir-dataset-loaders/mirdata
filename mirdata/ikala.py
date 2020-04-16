# -*- coding: utf-8 -*-
"""iKala Dataset Loader

The iKala dataset is comprised of 252 30-second excerpts sampled from 206 iKala
songs (plus 100 hidden excerpts reserved for MIREX).
The music accompaniment and the singing voice are recorded at the left and right
channels respectively and can be found under the Wavfile directory.
In addition, the human-labeled pitch contours and timestamped lyrics can be
found under PitchLabel and Lyrics respectively.

For more details, please visit: http://mac.citi.sinica.edu.tw/ikala/
"""

import csv
import os
import librosa
import logging
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils


DATASET_DIR = 'iKala'
TIME_STEP = 0.032  # seconds
REMOTES = {
    'metadata': download_utils.RemoteFileMetadata(
        filename='id_mapping.txt',
        url='http://mac.citi.sinica.edu.tw/ikala/id_mapping.txt',
        checksum='81097b587804ce93e56c7a331ba06abc',
        destination_dir=None,
    )
}


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


class Track(track.Track):
    """ikala Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_path (str): path to the track's audio file
        f0_path (str): path to the track's f0 annotation file
        lyrics_path (str): path to the track's lyric annotation file
        section (str): section. Either 'verse' or 'chorus'
        singer_id (str): singer id
        song_id (str): song id of the track
        track_id (str): track id

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
        self.f0_path = os.path.join(self._data_home, self._track_paths['pitch'][0])
        self.lyrics_path = os.path.join(self._data_home, self._track_paths['lyrics'][0])

        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])
        self.song_id = track_id.split('_')[0]
        self.section = track_id.split('_')[1]

        if metadata is not None and self.song_id in metadata:
            self.singer_id = metadata[self.song_id]
        else:
            self.singer_id = None

    @utils.cached_property
    def f0(self):
        """F0Data: The human-annotated singing voice pitch"""
        return load_f0(self.f0_path)

    @utils.cached_property
    def lyrics(self):
        """LyricData: The human-annotated lyrics"""
        return load_lyrics(self.lyrics_path)

    @property
    def vocal_audio(self):
        """(np.ndarray, float): mono vocal audio signal, sample rate"""
        return load_vocal_audio(self.audio_path)

    @property
    def instrumental_audio(self):
        """(np.ndarray, float): mono instrumental audio signal, sample rate"""
        return load_instrumental_audio(self.audio_path)

    @property
    def mix_audio(self):
        """(np.ndarray, float): mono mixture audio signal, sample rate"""
        return load_mix_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.f0, None)],
            lyrics_data=[(self.lyrics, None)],
            metadata={
                'section': self.section,
                'singer_id': self.singer_id,
                'track_id': self.track_id,
                'song_id': self.song_id,
            },
        )


def load_vocal_audio(audio_path):
    """Load an ikala vocal.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    vocal_channel = audio[1, :]
    return vocal_channel, sr


def load_instrumental_audio(audio_path):
    """Load an ikala instrumental.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    instrumental_channel = audio[0, :]
    return instrumental_channel, sr


def load_mix_audio(audio_path):
    """Load an ikala mix.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    mixed_audio, sr = librosa.load(audio_path, sr=None, mono=True)
    # multipy by 2 because librosa averages the left and right channel.
    return 2.0 * mixed_audio, sr


def download(data_home=None, force_overwrite=False):
    """Download iKala Dataset. However, iKala dataset is not available for
    download anymore. This function prints a helper message to organize
    pre-downloaded iKala dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
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
        remotes=REMOTES,
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


def load_f0(f0_path):
    if not os.path.exists(f0_path):
        raise IOError("f0_path {} does not exist".format(f0_path))

    with open(f0_path) as fhandle:
        lines = fhandle.readlines()
    f0_midi = np.array([float(line) for line in lines])
    f0_hz = librosa.midi_to_hz(f0_midi) * (f0_midi > 0)
    confidence = (f0_hz > 0).astype(float)
    times = (np.arange(len(f0_midi)) * TIME_STEP) + (TIME_STEP / 2.0)
    f0_data = utils.F0Data(times, f0_hz, confidence)
    return f0_data


def load_lyrics(lyrics_path):
    if not os.path.exists(lyrics_path):
        raise IOError("lyrics_path {} does not exist".format(lyrics_path))

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
