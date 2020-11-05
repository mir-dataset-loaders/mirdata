# -*- coding: utf-8 -*-
"""Saraga Dataset Loader

This repository contains time aligned melody, rhythm and structural annotations for two large open corpora of
Indian Art Music (Carnatic and Hindustani music).

The repository contains the following manual annotations referring to audio files:

Section and tempo annotations stored as start and end timestamps together with the name of the section and
tempo during the section (in a separate file). Sama annotations referring to rhythmic cycle boundaries stored
as timestamps. Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
({S, r, R, g, G, m, M, P, d, D, n, N}). Audio features automatically extracted and stored: pitch and tonic.
"""

import librosa
import numpy as np
import os
import json
import logging

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import utils


BIBTEX = """
@dataset{bozkurt_b_2018_1256127,
  author       = {Bozkurt, B. and
                  Srinivasamurthy, A. and
                  Gulati, S. and
                  Serra, X.},
  title        = {Saraga: research datasets of Indian Art Music},
  month        = may,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.1256127},
  url          = {https://doi.org/10.5281/zenodo.1256127}
}
"""


REMOTES = {
    'all': download_utils.RemoteFileMetadata(
        filename='saraga_1.0.zip',
        url='https://zenodo.org/record/1256127/files/saraga_1.0.zip?download=1',
        checksum='c8471e55bd55e060bde6cfacc555e1b1',
        destination_dir=None,
    )
}


def _load_metadata(metadata_path):

    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)
        data_home = metadata_path.split('/' + metadata_path.split('/')[-3])[0]
        metadata['track_id'] = str(metadata_path.split('/')[-3]) + '_' + str(metadata_path.split('/')[-2])
        metadata['data_home'] = data_home

        return metadata


DATA = utils.LargeData('saraga_index.json', _load_metadata)


class Track(core.Track):
    """Saraga Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        TODO
    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index:
            raise ValueError('{} is not a valid track ID in Salami'.format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]

        # Audio path
        self.audio_path = os.path.join(self._data_home, self._track_paths['audio'][0])

        # Annotation paths
        self.ctonic_path = utils.none_path_join(
            [self._data_home, self._track_paths['ctonic'][0]]
        )
        self.pitch_path = utils.none_path_join(
            [self._data_home, self._track_paths['pitch'][0]]
        )
        self.pitch_vocal_path = utils.none_path_join(
            [self._data_home, self._track_paths['pitch_vocal'][0]]
        )
        self.bpm_path = utils.none_path_join(
            [self._data_home, self._track_paths['bpm'][0]]
        )
        self.tempo_path = utils.none_path_join(
            [self._data_home, self._track_paths['tempo'][0]]
        )
        self.sama_path = utils.none_path_join(
            [self._data_home, self._track_paths['sama'][0]]
        )
        self.sections_path = utils.none_path_join(
            [self._data_home, self._track_paths['sections'][0]]
        )
        self.phrases_path = utils.none_path_join(
            [self._data_home, self._track_paths['phrases'][0]]
        )
        self.metadata_path = utils.none_path_join(
            [self._data_home, self._track_paths['metadata'][0]]
        )

        # Flag to separate between carnatinc and hindustani tracks
        self.iam_style = str(self.track_id.split('_')[0])

        # CARNATIC MUSIC TRACKS
        if self.iam_style == 'carnatic':
            metadata = DATA.metadata(self.metadata_path)
            if metadata is not None and track_id == metadata['track_id']:
                self._track_metadata = metadata
            else:
                # annotations with missing metadata
                self._track_metadata = {
                    'raaga': None,
                    'form': None,
                    'title': None,
                    'work': None,
                    'length': None,
                    'taala': None,
                    'album_artists': None,
                    'mbid': None,
                    'artists': None,
                    'concert': None
                }

            self.raaga = self._track_metadata['raaga']
            self.form = self._track_metadata['form']
            self.title = self._track_metadata['title']
            self.work = self._track_metadata['work']
            self.taala = self._track_metadata['taala']
            self.album_artists = self._track_metadata['album_artists']
            self.mbid = self._track_metadata['mbid']
            self.artists = self._track_metadata['artists']
            self.concert = self._track_metadata['concert']

        # HINDUSTANI MUSIC TRACKS
        if self.iam_style == 'hindustani':
            metadata = DATA.metadata(self.metadata_path)
            if metadata is not None and track_id == metadata['track_id']:
                self._track_metadata = metadata
            else:
                # annotations with missing metadata
                self._track_metadata = {
                    'title': None,
                    'raags': None,
                    'length': None,
                    'album_artists': None,
                    'forms': None,
                    'mbid': None,
                    'artists': None,
                    'release': None,
                    'works': None,
                    'taals': None,
                    'layas': None
                }

            self.title = self._track_metadata['title']
            self.raags = self._track_metadata['raags']
            self.album_artists = self._track_metadata['album_artists']
            self.forms = self._track_metadata['forms']
            self.mbid = self._track_metadata['mbid']
            self.artists = self._track_metadata['artists']
            self.release = self._track_metadata['release']
            self.works = self._track_metadata['works']
            self.taals = self._track_metadata['taals']
            self.layas = self._track_metadata['layas']

    @utils.cached_property
    def tonic(self):
        """Integer: tonic annotation"""
        if self.ctonic_path is None:
            return None
        return load_tonic(self.ctonic_path)

    @utils.cached_property
    def pitch(self):
        """F0Data: pitch annotation"""
        if self.pitch_path is None:
            return None
        return load_pitch(self.pitch_path)

    @utils.cached_property
    def pitch_vocal(self):
        """F0Data: pitch vocal annotations"""
        if self.pitch_vocal_path is None:
            return None
        return load_pitch(self.pitch_vocal_path)

    @utils.cached_property
    def bpm(self):
        """TempoData: tempo annotations"""
        if self.bpm_path is None:
            return None
        return load_bpm(self.bpm_path)

    '''
    @utils.cached_property
    def tempo(self):
        """TempoData: tempo annotations"""
        if self.tempo_path is None:
            return None
        return load_tempo(self.tempo_path)
    '''

    @utils.cached_property
    def sama(self):
        """SectionData: annotations in hierarchy level 1 from annotator 2"""
        if self.sama_path is None:
            return None
        return load_sama(self.sama_path)

    @utils.cached_property
    def sections(self):
        """SectionData: annotations in hierarchy level 1 from annotator 2"""
        if self.sections_path is None:
            return None
        return load_sections(self.sections_path)

    @utils.cached_property
    def phrases(self):
        """SectionData: annotations in hierarchy level 1 from annotator 2"""
        if self.phrases_path is None:
            return None
        return load_phrases(self.phrases_path)

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.pitch, 'pitch'), (self.pitch_vocal, 'pitch_vocal')],
            # tempo_data=[(self.bpm, 'bpm tempo')],
            section_data=[(self.sama, 'sama'), (self.sections, 'sections')],
            event_data=[(self.phrases, 'phrases')],
            metadata={
                'tonic': self.tonic,
                'metadata': self._track_metadata
            }
        )


def load_audio(audio_path):
    """Load a Saraga audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))

    return librosa.load(audio_path, sr=44100, mono=False)


def load_tonic(tonic_path):
    """Load tonic

    Args:
        tonic_path (str): Local path where the tonic path is stored.
            If `None`, returns None.

    Returns:
        (int): Tonic annotation in Hz
    """
    if tonic_path is None:
        return None

    if not os.path.exists(tonic_path):
        raise IOError("tonic_path {} does not exist".format(tonic_path))

    with open(tonic_path, 'r') as reader:
        return float(reader.readline().split('\n')[0])


def load_pitch(pitch_path):
    """Load pitch

    Args:
        pitch path (str): Local path where the pitch annotation is stored.
            If `None`, returns None.

    Returns:
        F0Data: pitch annotation
    """
    if pitch_path is None:
        return None

    if not os.path.exists(pitch_path):
        raise IOError("melody_path {} does not exist".format(pitch_path))

    times = []
    freqs = []
    with open(pitch_path, 'r') as reader:
        for line in reader.readlines():
            times.append(float(line.split('\t')[0]))
            freqs.append(float(line.split('\t')[1]))

    if not times:
        return None

    times = np.array(times)
    freqs = np.array(freqs)
    confidence = (freqs > 0).astype(float)
    return utils.F0Data(times, freqs, confidence)


def load_bpm(bpm_path):
    """Load bpm tempo

    Args:
        bpm_path (str): Local path where the bpm tempo is stored.
            If `None`, returns None.

    Returns:
        TempoData: bpm tempo annotation
    """
    if not os.path.exists(bpm_path):
        raise IOError("bpm_path {} does not exist".format(bpm_path))

    tempo = []
    start_times = []
    duration = []
    confidence = []
    with open(bpm_path, 'r') as reader:
        for line in reader.readlines():
            tempo.append(float(line.split(',')[0]))
            start_times.append(float(line.split(',')[1]))
            duration.append(float(line.split(',')[2]) - float(line.split(',')[1]))
            confidence.append(1.) if line.split(',')[0] is not None else confidence.append(0.)

    if not tempo:
        return None

    return utils.TempoData(
        np.array(start_times),
        np.array(duration),
        np.array(tempo),
        np.array(confidence)
    )


'''
def load_tempo(tempo_path):
    """Load tempo

    Args:
        tempo_path (str): Local path where the tempo annotation is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Returns:
        (dict): {`track_id`: track data}

    """
'''


def load_sama(sama_path):
    """Load sama

    Args:
        sama_path (str): Local path where the sama annotation is stored.
            If `None`, returns None.

    Returns:
        SectionData: sama annotations

    """
    if sama_path is None:
        return None

    if not os.path.exists(sama_path):
        raise IOError("sama_path {} does not exist".format(sama_path))

    timestamps = []
    sama_cycles = []
    intervals = []
    with open(sama_path, 'r') as reader:
        for line in reader.readlines():
            timestamps.append(float(line))

    for i in np.arange(1, len(timestamps)):
        intervals.append([timestamps[i-1], timestamps[i]])
        sama_cycles.append('sama cycle ' + str(i))

    if not intervals:
        return None

    return utils.SectionData(
            np.array(intervals),
            sama_cycles
        )


def load_sections(sections_path):
    """Load sections

    Args:
        sections_path (str): Local path where the section annotation is stored.
            If `None`, returns None.

    Returns:
        SectionData: section annotations for track

    """
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    intervals = []
    section_labels = []
    with open(sections_path, 'r') as reader:
        for line in reader.readlines():
            intervals.append([float(line.split('\t')[0]), float(line.split('\t')[0]) + float(line.split('\t')[2])])
            section_labels.append(str(line.split('\t')[3].split('\n')[0]) + '_' + str(line.split('\t')[1]))

    if not intervals:
        return None

    return utils.SectionData(
        np.array(intervals),
        section_labels
    )


def load_phrases(phrases_path):
    """Load phrases

    Args:
        phrases_path (str): Local path where the phrase annotation is stored.
            If `None`, returns None.

    Returns:
        EventData: phrases annotation for track

    """
    if phrases_path is None:
        return None

    if not os.path.exists(phrases_path):
        raise IOError("sections_path {} does not exist".format(phrases_path))

    start_times = []
    end_times = []
    events = []
    with open(phrases_path, 'r') as reader:
        for line in reader.readlines():
            start_times.append(float(line.split('\t')[0]))
            end_times.append(float(line.split('\t')[0]) + float(line.split('\t')[2]))
            events.append(str(line.split('\t')[3].split('\n')[0]) + '_' + str(line.split('\t')[1]))

    if not start_times:
        return None

    return utils.EventData(
        np.array(start_times),
        np.array(end_times),
        events
    )
