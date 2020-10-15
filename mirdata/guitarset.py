# -*- coding: utf-8 -*-
"""GuitarSet Loader

GuitarSet provides audio recordings of a variety of musical excerpts
played on an acoustic guitar, along with time-aligned annotations
including pitch contours, string and fret positions, chords, beats,
downbeats, and keys.

GuitarSet contains 360 excerpts that are close to 30 seconds in length.
The 360 excerpts are the result of the following combinations:

- 6 players
- 2 versions: comping (harmonic accompaniment) and soloing (melodic improvisation)
- 5 styles: Rock, Singer-Songwriter, Bossa Nova, Jazz, and Funk
- 3 Progressions: 12 Bar Blues, Autumn Leaves, and Pachelbel Canon.
- 2 Tempi: slow and fast.

The tonality (key) of each excerpt is sampled uniformly at random.

GuitarSet was recorded with the help of a hexaphonic pickup, which outputs
signals for each string separately, allowing automated note-level annotation.
Excerpts are recorded with both the hexaphonic pickup and a Neumann U-87
condenser microphone as reference.
3 audio recordings are provided with each excerpt with the following suffix:

- hex: original 6 channel wave file from hexaphonic pickup
- hex_cln: hex wave files with interference removal applied
- mic: monophonic recording from reference microphone
- mix: monophonic mixture of original 6 channel file

Each of the 360 excerpts has an accompanying JAMS file which stores 16 annotations.
Pitch:

- 6 pitch_contour annotations (1 per string)
- 6 midi_note annotations (1 per string)

Beat and Tempo:

- 1 beat_position annotation
- 1 tempo annotation

Chords:

- 2 chord annotations: instructed and performed. The instructed chord annotation
is a digital version of the lead sheet that's provided to the player, and the
performed chord annotations are inferred from note annotations, using
segmentation and root from the digital lead sheet annotation.

For more details, please visit: http://github.com/marl/guitarset/
"""

import jams
import librosa
import logging
import numpy as np
import os

from mirdata import download_utils
from mirdata import track
from mirdata import utils


DATASET_DIR = 'GuitarSet'

REMOTES = {
    'annotations': download_utils.RemoteFileMetadata(
        filename='annotation.zip',
        url='https://zenodo.org/record/3371780/files/annotation.zip?download=1',
        checksum='b39b78e63d3446f2e54ddb7a54df9b10',
        destination_dir='annotation',
    ),
    'audio_hex_debleeded': download_utils.RemoteFileMetadata(
        filename='audio_hex-pickup_debleeded.zip',
        url='https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1',
        checksum='c31d97279464c9a67e640cb9061fb0c6',
        destination_dir='audio_hex-pickup_debleeded',
    ),
    'audio_hex_original': download_utils.RemoteFileMetadata(
        filename='audio_hex-pickup_original.zip',
        url='https://zenodo.org/record/3371780/files/audio_hex-pickup_original.zip?download=1',
        checksum='f9911bf217cb40e9e68edf3726ef86cc',
        destination_dir='audio_hex-pickup_original',
    ),
    'audio_mic': download_utils.RemoteFileMetadata(
        filename='audio_mono-mic.zip',
        url='https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1',
        checksum='275966d6610ac34999b58426beb119c3',
        destination_dir='audio_mono-mic',
    ),
    'audio_mix': download_utils.RemoteFileMetadata(
        filename='audio_mono-pickup_mix.zip',
        url='https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip?download=1',
        checksum='aecce79f425a44e2055e46f680e10f6a',
        destination_dir='audio_mono-pickup_mix',
    ),
}
_STYLE_DICT = {
    'Jazz': 'Jazz',
    'BN': 'Bossa Nova',
    'Rock': 'Rock',
    'SS': 'Singer-Songwriter',
    'Funk': 'Funk',
}
_GUITAR_STRINGS = ['E', 'A', 'D', 'G', 'B', 'e']
DATA = utils.LargeData('guitarset_index.json')


class Track(track.Track):
    """guitarset Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        audio_hex_cln_path (str): path to the debleeded hex wave file
        audio_hex_path (str): path to the original hex wave file
        audio_mic_path (str): path to the mono wave via microphone
        audio_mix_path (str): path to the mono wave via downmixing hex pickup
        jams_path (str): path to the jams file
        mode (str): one of ['solo', 'comp']
            For each excerpt, players are asked to first play in 'comp' mode
            and later play a 'solo' version on top of the already recorded comp.
        player_id (str): ID of the different players.
            one of ['00', '01', ... , '05']
        style (str): one of ['Jazz', 'Bossa Nova', 'Rock', 'Singer-Songwriter', 'Funk']
        tempo (float): BPM of the track
        track_id (str): track id

    """

    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in GuitarSet'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        self.audio_hex_cln_path = os.path.join(
            self._data_home, self._track_paths['audio_hex_cln'][0]
        )
        self.audio_hex_path = os.path.join(
            self._data_home, self._track_paths['audio_hex'][0]
        )
        self.audio_mic_path = os.path.join(
            self._data_home, self._track_paths['audio_mic'][0]
        )
        self.audio_mix_path = os.path.join(
            self._data_home, self._track_paths['audio_mix'][0]
        )
        self.jams_path = os.path.join(self._data_home, self._track_paths['jams'][0])

        title_list = track_id.split('_')  # [PID, S-T-K, mode, rec_mode]
        style, tempo, _ = title_list[1].split('-')  # [style, tempo, key]
        self.player_id = title_list[0]
        self.mode = title_list[2]
        self.tempo = float(tempo)
        self.style = _STYLE_DICT[style[:-1]]

    @utils.cached_property
    def beats(self):
        """BeatData: the track's beat positions"""
        return load_beats(self.jams_path)

    @utils.cached_property
    def leadsheet_chords(self):
        """ChordData: the track's chords as written in the leadsheet"""
        if self.mode == 'solo':
            logging.info(
                'Chord annotations for solo excerpts are the same with the comp excerpt.'
            )
        return load_chords(self.jams_path, leadsheet_version=True)

    @utils.cached_property
    def inferred_chords(self):
        """ChordData: the track's chords inferred from played transcription"""
        if self.mode == 'solo':
            logging.info(
                'Chord annotations for solo excerpts are the same with the comp excerpt.'
            )
        return load_chords(self.jams_path, leadsheet_version=False)

    @utils.cached_property
    def key_mode(self):
        """KeyData: the track's key and mode"""
        return load_key_mode(self.jams_path)

    @utils.cached_property
    def pitch_contours(self):
        """(dict): a dict that contains 6 F0Data.
        From Low E string to high e string.
        {
            'E': F0Data(...),
            'A': F0Data(...),
            ...
            'e': F0Data(...)
        }
        """
        contours = {}
        # iterate over 6 strings
        for i in range(6):
            contours[_GUITAR_STRINGS[i]] = load_pitch_contour(self.jams_path, i)
        return contours

    @utils.cached_property
    def notes(self):
        """dict: a dict that contains 6 NoteData.
        From Low E string to high e string.
        {
            'E': NoteData(...),
            'A': NoteData(...),
            ...
            'e': NoteData(...)
        }
        """
        notes = {}
        # iterate over 6 strings
        for i in range(6):
            notes[_GUITAR_STRINGS[i]] = load_note_ann(self.jams_path, i)
        return notes

    @property
    def audio_mic(self):
        """(np.ndarray, float): stereo microphone audio signal, sample rate"""
        audio, sr = load_audio(self.audio_mic_path)
        return audio, sr

    @property
    def audio_mix(self):
        """(np.ndarray, float): stereo mix audio signal, sample rate"""
        audio, sr = load_audio(self.audio_mix_path)
        return audio, sr

    @property
    def audio_hex(self):
        """(np.ndarray, float): raw hexaphonic audio signal, sample rate"""
        audio, sr = load_multitrack_audio(self.audio_hex_path)
        return audio, sr

    @property
    def audio_hex_cln(self):
        """(np.ndarray, float): bleed-removed hexaphonic audio signal, sample rate"""
        audio, sr = load_multitrack_audio(self.audio_hex_cln_path)
        return audio, sr

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams.load(self.jams_path)


def load_audio(audio_path):
    """Load a Guitarset audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)


def load_multitrack_audio(audio_path):
    """Load a Guitarset multitrack audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=False)


def download(
    data_home=None, partial_download=None, force_overwrite=False, cleanup=True
):
    """Download GuitarSet.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        partial_download (list):
            List indicating what to partially download. The list can include any of:
             * `'annotations'` the annotation files
             * `'audio_hex_original'` original 6 channel wave file from hexaphonic pickup
             * `'audio_hex_debleeded'` hex wave files with interference removal applied
             * `'audio_mic'` monophonic recording from reference microphone
             * `'audio_mix'` monophonic mixture of original 6 channel file
             If `None`, all data is downloaded.
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        data_home,
        remotes=REMOTES,
        partial_download=partial_download,
        info_message=None,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
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
    """Load GuitarSet

    Args:
        data_home (str): Local path where GuitarSet is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    guitarset_data = {}
    for key in DATA.index.keys():
        guitarset_data[key] = Track(key, data_home=data_home)
    return guitarset_data


def load_beats(jams_path):
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    anno = jam.search(namespace='beat_position')[0]
    times, values = anno.to_event_values()
    positions = [int(v['position']) for v in values]
    return utils.BeatData(times, positions)


def load_chords(jams_path, leadsheet_version=True):
    """
    Args:
        jams_path (str): Path of the jams annotation file
        leadsheet_version (Bool)
            Whether or not to load the leadsheet version of the chord annotation
            If False, load the infered version.

    Returns:
        (ChordData): Chord data
    """
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    if leadsheet_version:
        anno = jam.search(namespace='chord')[0]
    else:
        anno = jam.search(namespace='chord')[1]
    intervals, values = anno.to_interval_values()
    return utils.ChordData(intervals, values)


def load_key_mode(jams_path):
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    anno = jam.search(namespace='key_mode')[0]
    intervals, values = anno.to_interval_values()
    return utils.KeyData(intervals[:, 0], intervals[:, 1], values)


def load_pitch_contour(jams_path, string_num):
    """
    Args:
        jams_path (str): Path of the jams annotation file

    string_num (int), in range(6): Which string to load.
        0 is the Low E string, 5 is the high e string.
    """
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    anno_arr = jam.search(namespace='pitch_contour')
    anno = anno_arr.search(data_source=str(string_num))[0]
    times, values = anno.to_event_values()
    frequencies = [v['frequency'] for v in values]
    return utils.F0Data(times, frequencies, np.ones_like(times))


def load_note_ann(jams_path, string_num):
    """
    Args:
        jams_path (str): Path of the jams annotation file

    string_num (int), in range(6): Which string to load.
        0 is the Low E string, 5 is the high e string.
    """
    if not os.path.exists(jams_path):
        raise IOError("jams_path {} does not exist".format(jams_path))
    jam = jams.load(jams_path)
    anno_arr = jam.search(namespace='note_midi')
    anno = anno_arr.search(data_source=str(string_num))[0]
    intervals, values = anno.to_interval_values()
    return utils.NoteData(intervals, values, np.ones_like(values))


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Xi, Qingyang, et al.
"GuitarSet: A Dataset for Guitar Transcription."
In Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR). 2018.
========== Bibtex ==========
@inproceedings{xi2018guitarset,
    title={GuitarSet: A Dataset for Guitar Transcription},
    author={Xi, Qingyang and Bittner, Rachel M and Ye, Xuzhou and Pauwels, Johan and Bello, Juan P},
    booktitle={International Society of Music Information Retrieval (ISMIR)},
    year={2018}
}
"""
    print(cite_data)
