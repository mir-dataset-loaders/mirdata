# -*- coding: utf-8 -*-
"""Saraga Carnatic Multitrack Dataset Loader

This repository contains time aligned melody, rhythm and structural annotations for a large open corpora of
Carnatic music, as well as multitrack audio comprising isolated audio tracks for the following instruments:
Ghatam, mridangam (right and left channel), violin, vocal secondary and main vocal.

Section and tempo annotations stored as start and end timestamps together with the name of the section and
tempo during the section. Sama annotations referring to rhythmic cycle boundaries stored
as timestamps. Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
({S, r, R, g, G, m, M, P, d, D, n, N}). Audio features automatically extracted and stored: pitch and tonic.
The annotations are stored in text files, named as the audio filename but with the respective extension at the
end, for instance: "Bhuvini Dasudane.tempo-manual.txt".

This dataset has a total of 168 tracks from the carnatic dataset have multitrack audio.
Count of annotations for the total 168 tracks:
    'audio': 168
    'audio-ghatam': 46
    'audio-mridangam-left': 168
    'audio-mridangam-right': 168
    'audio-violin': 168
    'audio-vocal-s': 24
    'audio-vocal': 168
    'ctonic': 116
    'pitch': 116
    'phrases': 45
    'tempo': 60
    'sama': 63
    'sections': 46
    'metadata': 168

The files of this dataset are shared with the following license:
Creative Commons Attribution Non Commercial Share Alike 4.0 International

Dataset compiled by: Bozkurt, B.; Srinivasamurthy, A.; Gulati, S. and Serra, X.

For more information about the dataset as well as IAM and annotations, please refer to:
https://mtg.github.io/saraga/, where a really detailed explanation of the data and annotations is published.
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
        filename='saraga_multitrack1.0.zip',
        url='TODO',
        checksum='TODO',
        destination_dir=None,
    )
}

MULTITRACK_DICT = {
    'audio-ghatam',
    'audio-mridangam-left',
    'audio-mridangam-right',
    'audio-violin',
    'audio-vocal-s',
    'audio-vocal'
}


def _load_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        logging.info('Metadata file {} not found.'.format(metadata_path))
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)
        data_home = metadata_path.split('/' + metadata_path.split('/')[-3])[0]
        metadata['data_home'] = data_home

        return metadata


DATA = utils.LargeData('saraga_multitrack_index.json', _load_metadata)


class Track(core.MultiTrack):
    """Saraga Track class

         Args:
             mtrack_id (str): track id of the track
             data_home (str): Local path where the dataset is stored. default=None
                 If `None`, looks for the data in the default directory, `~/mir_datasets`

         Attributes:
             title (str): Title of the piece in the track
             mbid (str): MusicBrainz ID of the track
             album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
             artists (list, dicts): list of dicts containing information of the featuring artists in the track
             raaga (list, dict): list of dicts containing information about the raagas present in the track
             form (list, dict): list of dicts containing information about the forms present in the track
             work (list, dicts): list of dicts containing the work present in the piece, and its mbid
             taala (list, dicts): list of dicts containing the talas present in the track and its uuid
             concert (list, dicts): list of dicts containing the concert where the track is present and its mbid

             audio_path (str): path of the audio file of the mix track
             multitrack_ids (list, str): list containing ids of available multitrack single instrument files
             multitrack_paths (list, str): list containing paths of available multitrack single instrument files

    """

    def __init__(self, mtrack_id, data_home):
        if mtrack_id not in DATA.index['tracks']:
            raise ValueError('{} is not a valid track ID in Saraga Multitrack'.format(mtrack_id))

        self.mtrack_id = mtrack_id

        self._data_home = data_home
        self.audio_path = os.path.join(data_home, DATA.index['tracks'][mtrack_id]['audio'][0])

        self._data_home = data_home
        self._track_paths = DATA.index['tracks'][mtrack_id]

        self.multitrack_ids = [
            k for k, v in sorted(DATA.index['multitracks'][mtrack_id].items()) if v != [None, None]
        ]

        self.multitrack_paths = []
        # Audio paths of multitracks
        for i in self.multitrack_ids:
            assert (i in MULTITRACK_DICT), "Multitrack file {} not in multitrack dictionary".format(i)
            self.multitrack_paths.append(os.path.join(data_home, DATA.index['multitracks'][mtrack_id][i][0]))

        # Annotation paths
        self.ctonic_path = utils.none_path_join(
            [self._data_home, self._track_paths['ctonic'][0]]
        )
        self.pitch_path = utils.none_path_join(
            [self._data_home, self._track_paths['pitch'][0]]
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

        # CARNATIC MUSIC TRACKS
        metadata = DATA.metadata(self.metadata_path)
        if metadata is not None and mtrack_id.split('_')[1] in metadata['title']:
            metadata['mtrack_id'] = mtrack_id
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

        self.title = self._track_metadata['title'] if 'title' in self._track_metadata.keys() is not None else None
        self.artists = self._track_metadata['artists'] if 'artists' in self._track_metadata.keys() is not None else None
        self.album_artists = self._track_metadata['album_artists'] if 'album_artists' in self._track_metadata.keys() is not None else None
        self.mbid = self._track_metadata['mbid'] if 'mbid' in self._track_metadata.keys() is not None else None
        self.raaga = self._track_metadata['raaga'] if 'raaga' in self._track_metadata.keys() is not None else None
        self.form = self._track_metadata['form'] if 'form' in self._track_metadata.keys() is not None else None
        self.work = self._track_metadata['work'] if 'work' in self._track_metadata.keys() is not None else None
        self.taala = self._track_metadata['taala'] if 'taala' in self._track_metadata.keys() is not None else None
        self.concert = self._track_metadata['concert'] if 'concert' in self._track_metadata.keys() is not None else None

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)

    @property
    def multitrack_audio(self):
        """(dict, Track): audio signal, sample rate"""
        return {
            k: SingleTrack(self.mtrack_id, k, self._data_home) for k in self.multitrack_ids
        }

    @utils.cached_property
    def tonic(self):
        """Float: tonic annotation"""
        return load_tonic(self.ctonic_path)

    @utils.cached_property
    def pitch(self):
        """F0Data: pitch annotation"""
        return load_pitch(self.pitch_path)

    @utils.cached_property
    def tempo(self):
        """Dict: tempo annotations"""
        return load_tempo(self.tempo_path)

    @utils.cached_property
    def sama(self):
        """SectionData: sama section annotations"""
        return load_sama(self.sama_path)

    @utils.cached_property
    def sections(self):
        """SectionData: sama section annotations"""
        return load_sections(self.sections_path)

    @utils.cached_property
    def phrases(self):
        """SectionData: sama section annotations"""
        return load_phrases(self.phrases_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            f0_data=[(self.pitch, 'pitch')],
            section_data=[(self.sama, 'sama'), (self.sections, 'sections')],
            event_data=[(self.phrases, 'phrases')],
            metadata={
                'tempo': self.tempo,
                'tonic': self.tonic,
                'metadata': self._track_metadata
            }
        )


class SingleTrack(core.Track):
    """Saraga Single Track class

         Args:
             mtrack_id (str): track id of the mix track
             strack_id (str): track id of the single instrument track
             data_home (str): Local path where the dataset is stored. default=None
                 If `None`, looks for the data in the default directory, `~/mir_datasets`

         Attributes:
             audio_path (str): path of the audio file of the single instrument track

    """
    def __init__(self, mtrack_id, strack_id, data_home):
        if mtrack_id not in DATA.index['tracks']:
            raise ValueError('{} is not a valid track ID in Saraga Multitrack'.format(mtrack_id))

        if strack_id not in DATA.index['multitracks'][mtrack_id]:
            raise ValueError('{} is not a valid multitrack ID in Saraga Multitrack'.format(strack_id))

        self.mtrack_id = mtrack_id  # Id for the general mix of the piece
        self.strack_id = strack_id  # Id for the single instrument track

        self._data_home = data_home
        assert (self.strack_id in MULTITRACK_DICT), "Multitrack file {} not in multitrack dictionary".format(self.strack_id)
        if DATA.index['multitracks'][mtrack_id][strack_id][0] is not None:
            self.audio_path = os.path.join(data_home, DATA.index['multitracks'][mtrack_id][strack_id][0])
        else:
            self.audio_path = None

    @property
    def audio(self):
        """(np.ndarray, float): audio signal, sample rate"""
        return load_audio(self.audio_path)


def load_audio(audio_path):
    """Load a Saraga audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if audio_path is None:
        return None

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


def load_tempo(tempo_path):
    """Load tempo from carnatic collection

    Args:
        tempo_path (str): Local path where the tempo annotation is stored.

    Returns:
        if carnatic:
        (dict): {'tempo_apm': tempo in aksharas per minute (APM)
                 'tempo_bpm': tempo in beats per minute (BPM)
                 'sama_interval': median duration (in seconds) of one tāla cycle
                 'beats_per_cycle': number of beats in one cycle of the tāla
                 'subdivisions': number of aksharas per beat of the tāla
                 }
    """
    if tempo_path is None:
        return None

    if not os.path.exists(tempo_path):
        raise IOError("tempo_path {} does not exist".format(tempo_path))

    tempo_annotation = {}

    with open(tempo_path, 'r') as reader:
        parsed_tempo = reader.readline()

    tempo_data = []
    tempo_apm = parsed_tempo.split(',')[0]
    tempo_data.append(tempo_apm)
    tempo_bpm = parsed_tempo.split(',')[1].split(' ')[1]
    tempo_data.append(tempo_bpm)
    sama_interval = parsed_tempo.split(',')[2].split(' ')[1]
    tempo_data.append(sama_interval)
    beats_per_cycle = parsed_tempo.split(',')[3].split(' ')[1]
    tempo_data.append(beats_per_cycle)
    subdivisions = parsed_tempo.split(',')[4].split(' ')[1]
    tempo_data.append(subdivisions)

    if 'NaN' in tempo_data:
        return None

    tempo_annotation['tempo_apm'] = float(tempo_apm) if '.' in tempo_apm else int(tempo_apm)
    tempo_annotation['tempo_bpm'] = float(tempo_bpm) if '.' in tempo_bpm else int(tempo_bpm)
    tempo_annotation['sama_interval'] = float(sama_interval) if '.' in sama_interval else int(sama_interval)
    tempo_annotation['beats_per_cycle'] = float(beats_per_cycle) if '.' in beats_per_cycle else int(beats_per_cycle)
    tempo_annotation['subdivisions'] = float(subdivisions) if '.' in subdivisions else int(subdivisions)

    return tempo_annotation


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
        intervals.append([timestamps[i - 1], timestamps[i]])
        sama_cycles.append('sama cycle ' + str(i))

    if not intervals:
        return None

    return utils.SectionData(
        np.array(intervals),
        sama_cycles
    )


def load_sections(sections_path):
    """Load sections from carnatic collection

    Args:
        sections_path (str): Local path where the section annotation is stored.

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
            if line != '\n':
                intervals.append([float(line.split('\t')[0]), float(line.split('\t')[0]) + float(line.split('\t')[2])])
                section_labels.append(str(line.split('\t')[3].split('\n')[0]))

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
            if len(line.split('\t')) == 4:
                start_times.append(float(line.split('\t')[0]))
                end_times.append(float(line.split('\t')[0]) + float(line.split('\t')[2]))
                events.append(str(line.split('\t')[3].split('\n')[0]))
            if len(line.split('\t')) == 3:
                start_times.append(float(line.split('\t')[0]))
                end_times.append(float(line.split('\t')[0]) + float(line.split('\t')[2]))
                events.append('No information')

    if not start_times:
        return None

    return utils.EventData(
        np.array(start_times),
        np.array(end_times),
        events
    )
