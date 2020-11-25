# -*- coding: utf-8 -*-
"""Saraga Dataset Loader

This repository contains time aligned melody, rhythm and structural annotations for two large open corpora of
Indian Art Music (Carnatic and Hindustani music).

The repository contains the following manual annotations referring to audio files:

Section and tempo annotations stored as start and end timestamps together with the name of the section and
tempo during the section (in a separate file). Sama annotations referring to rhythmic cycle boundaries stored
as timestamps. Phrase annotations stored as timestamps and transcription of the phrases using solfège symbols
({S, r, R, g, G, m, M, P, d, D, n, N}). Audio features automatically extracted and stored: pitch and tonic.
The annotations are stored in text files, named as the audio filename but with the respective extension at the
end, for instance: "Cherthala Ranganatha Sharma - Bhuvini Dasudane.tempo-manual.txt".

The dataset contains a total of 197 tracks from the carnatic corpus and 108 track from the hindustani corpus.

A total of 163 tracks from the carnatic dataset have multitrack audio, which is not currently available in this
loader version. New version of the loader is coming soon with the multitrack audio support.

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
        metadata['track_id'] = (
            str(metadata_path.split('/')[-3]) + '_' + str(metadata_path.split('/')[-2])
        )
        metadata['data_home'] = data_home

        return metadata


DATA = utils.LargeData('saraga_index.json', _load_metadata)


class Track(core.Track):
    """Saraga Track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored. default=None
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Common attributes:
        iam_style (str): flag to identify if track belongs to hindustani or carnatic collection
        title (str): Title of the piece in the track
        mbid (str): MusicBrainz ID of the track
        album_artists (list, dicts): list of dicts containing the album artists present in the track and its mbid
        artists (list, dicts): list of dicts containing information of the featuring artists in the track

    Carnatic attributes:
        raaga (list, dict): list of dicts containing information about the raagas present in the track
        form (list, dict): list of dicts containing information about the forms present in the track
        work (list, dicts): list of dicts containing the work present in the piece, and its mbid
        taala (list, dicts): list of dicts containing the talas present in the track and its uuid
        concert (list, dicts): list of dicts containing the concert where the track is present and its mbid

    Hindustani attributes:
        raags (list, dict): list of dicts containing information about the raags present in the track
        form (list, dict): list of dicts containing information about the forms present in the track
        release (list, dicts): list of dicts containing information of the release where the track is found
        work (list, dicts): list of dicts containing the work present in the piece, and its mbid
        taals (list, dicts): list of dicts containing the taals present in the track and its uuid
        layas (list, dicts): list of dicts containing the layas present in the track and its uuid
    """

    def __init__(self, track_id, data_home):
        if track_id not in DATA.index['tracks']:
            raise ValueError('{} is not a valid track ID in Saraga'.format(track_id))

        self.track_id = track_id

        self._data_home = data_home
        self._track_paths = DATA.index['tracks'][track_id]

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
                    'concert': None,
                }

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
                    'layas': None,
                }

        # Common attributes for Hindustani and Carnatic tracks
        self.title = self._track_metadata['title']
        self.artists = self._track_metadata['artists']
        self.album_artists = self._track_metadata['album_artists']
        self.mbid = self._track_metadata['mbid']

        # Carnatic specific attributes
        self.raaga = (
            self._track_metadata['raaga']
            if 'raaga' in self._track_metadata.keys() is not None
            else None
        )
        self.form = (
            self._track_metadata['form']
            if 'form' in self._track_metadata.keys() is not None
            else None
        )
        self.work = (
            self._track_metadata['work']
            if 'work' in self._track_metadata.keys() is not None
            else None
        )
        self.taala = (
            self._track_metadata['taala']
            if 'taala' in self._track_metadata.keys() is not None
            else None
        )
        self.concert = (
            self._track_metadata['concert']
            if 'concert' in self._track_metadata.keys() is not None
            else None
        )

        # Hindustani specific attributes
        self.raags = (
            self._track_metadata['raags']
            if 'raags' in self._track_metadata.keys() is not None
            else None
        )
        self.forms = (
            self._track_metadata['forms']
            if 'forms' in self._track_metadata.keys() is not None
            else None
        )
        self.release = (
            self._track_metadata['release']
            if 'release' in self._track_metadata.keys() is not None
            else None
        )
        self.works = (
            self._track_metadata['works']
            if 'works' in self._track_metadata.keys() is not None
            else None
        )
        self.taals = (
            self._track_metadata['taals']
            if 'taals' in self._track_metadata.keys() is not None
            else None
        )
        self.layas = (
            self._track_metadata['layas']
            if 'layas' in self._track_metadata.keys() is not None
            else None
        )

    @utils.cached_property
    def tonic(self):
        """Float: tonic annotation"""
        return load_tonic(self.ctonic_path)

    @utils.cached_property
    def pitch(self):
        """F0Data: pitch annotation"""
        return load_pitch(self.pitch_path)

    @utils.cached_property
    def pitch_vocal(self):
        """F0Data: pitch vocal annotations"""
        return load_pitch(self.pitch_vocal_path)

    @utils.cached_property
    def tempo(self):
        """Dict: tempo annotations"""
        return load_tempo(self.tempo_path, self.iam_style)

    @utils.cached_property
    def sama(self):
        """SectionData: sama section annotations"""
        return load_sama(self.sama_path)

    @utils.cached_property
    def sections(self):
        """SectionData: track section annotations"""
        return load_sections(self.sections_path, self.iam_style)

    @utils.cached_property
    def phrases(self):
        """EventData: phrase annotations"""
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
            section_data=[(self.sama, 'sama'), (self.sections, 'sections')],
            event_data=[(self.phrases, 'phrases')],
            metadata={
                'tempo': self.tempo,
                'tonic': self.tonic,
                'metadata': self._track_metadata,
            },
        )


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


def load_tempo(tempo_path, iam_style):
    """Load tempo from carnatic collection

    Args:
        tempo_path (str): Local path where the tempo annotation is stored.
        iam_style (str): Flag to differentiate between hindustani and carnatic tracks

    Returns:
        if carnatic:
        (dict): {'tempo_apm': tempo in aksharas per minute (APM)
                 'tempo_bpm': tempo in beats per minute (BPM)
                 'sama_interval': median duration (in seconds) of one tāla cycle
                 'beats_per_cycle': number of beats in one cycle of the tāla
                 'subdivisions': number of aksharas per beat of the tāla
                 }

        if hindustani:
        (dict): {'tempo': median tempo for the section in mātrās per minute (MPM)
                 'matra_interval': tempo expressed as the duration of the mātra (essentially
                                   dividing 60 by tempo, expressed in seconds)
                 'sama_interval': median duration of one tāl cycle in the section
                 'matras_per_cycle': indicator of the structure of the tāl, showing the number
                                     of mātrā in a cycle of the tāl of the recording
                 'start_time': start time of the section
                 'duration': duration of the section
    """
    if tempo_path is None:
        return None

    if not os.path.exists(tempo_path):
        raise IOError("tempo_path {} does not exist".format(tempo_path))

    tempo_annotation = {}
    if iam_style == 'carnatic':

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

        tempo_annotation['tempo_apm'] = (
            float(tempo_apm) if '.' in tempo_apm else int(tempo_apm)
        )
        tempo_annotation['tempo_bpm'] = (
            float(tempo_bpm) if '.' in tempo_bpm else int(tempo_bpm)
        )
        tempo_annotation['sama_interval'] = (
            float(sama_interval) if '.' in sama_interval else int(sama_interval)
        )
        tempo_annotation['beats_per_cycle'] = (
            float(beats_per_cycle) if '.' in beats_per_cycle else int(beats_per_cycle)
        )
        tempo_annotation['subdivisions'] = (
            float(subdivisions) if '.' in subdivisions else int(subdivisions)
        )

    else:
        head, tail = os.path.split(tempo_path)
        sections_path = tail.split('.')[0] + '.sections-manual.txt'
        sections_abs_path = head + '/' + sections_path

        sections = []
        with open(sections_abs_path, 'r') as reader:
            for line in reader.readlines():
                if line != '\n':
                    sections.append(line.split(',')[3].split('\n')[0])

        section_count = 0
        with open(tempo_path, 'r') as reader:
            for line in reader.readlines():

                tempo_data = []
                tempo = line.split(',')[0]
                tempo_data.append(tempo)
                matra = line.split(',')[1].split(' ')[1]
                tempo_data.append(matra)
                sama_interval = line.split(',')[2].split(' ')[1]
                tempo_data.append(sama_interval)
                matras_per_cycle = line.split(',')[3].split(' ')[1]
                tempo_data.append(matras_per_cycle)
                start_time = line.split(',')[4].split(' ')[1]
                tempo_data.append(start_time)
                duration = line.split(',')[5].split(' ')[1]
                tempo_data.append(duration)

                if 'NaN' in tempo_data:
                    return None

                tempo_annotation[sections[section_count]] = {
                    'tempo': float(tempo) if '.' in tempo else int(tempo),
                    'matra_interval': float(matra) if '.' in matra else int(matra),
                    'sama_interval': float(sama_interval)
                    if '.' in sama_interval
                    else int(sama_interval),
                    'matras_per_cycle': float(matras_per_cycle)
                    if '.' in matras_per_cycle
                    else int(matras_per_cycle),
                    'start_time': float(start_time)
                    if '.' in start_time
                    else int(start_time),
                    'duration': float(duration) if '.' in duration else int(duration),
                }

                section_count += 1  # Go to next section

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

    return utils.SectionData(np.array(intervals), sama_cycles)


def load_sections(sections_path, iam_style):
    """Load sections from carnatic collection

    Args:
        sections_path (str): Local path where the section annotation is stored.
        iam_style (str): Flag to differentiate between hindustani and carnatic tracks

    Returns:
        SectionData: section annotations for track

    """
    if sections_path is None:
        return None

    if not os.path.exists(sections_path):
        raise IOError("sections_path {} does not exist".format(sections_path))

    intervals = []
    section_labels = []
    if iam_style == 'carnatic':
        with open(sections_path, 'r') as reader:
            for line in reader.readlines():
                if line != '\n':
                    intervals.append(
                        [
                            float(line.split('\t')[0]),
                            float(line.split('\t')[0]) + float(line.split('\t')[2]),
                        ]
                    )
                    section_labels.append(str(line.split('\t')[3].split('\n')[0]))

        if not intervals:
            return None

    if iam_style == 'hindustani':
        with open(sections_path, 'r') as reader:
            for line in reader.readlines():
                if line != '\n':
                    intervals.append(
                        [
                            float(line.split(',')[0]),
                            float(line.split(',')[0]) + float(line.split(',')[2]),
                        ]
                    )
                    section_labels.append(
                        str(line.split(',')[3].split('\n')[0])
                        + '-'
                        + str(line.split(',')[1])
                    )

        if not intervals:
            return None

    return utils.SectionData(np.array(intervals), section_labels)


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
                end_times.append(
                    float(line.split('\t')[0]) + float(line.split('\t')[2])
                )
                events.append(str(line.split('\t')[3].split('\n')[0]))
            if len(line.split('\t')) == 3:
                start_times.append(float(line.split('\t')[0]))
                end_times.append(
                    float(line.split('\t')[0]) + float(line.split('\t')[2])
                )
                events.append('No information')

    if not start_times:
        return None

    return utils.EventData(np.array(start_times), np.array(end_times), events)
