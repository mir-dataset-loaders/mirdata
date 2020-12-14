# -*- coding: utf-8 -*-
"""
Classes for the different annotation types in mirdata. Only annotations that are time series
are implemented here. Those that a single value per track are part of the metadata of a dataset
(e.g. constant tempo or key). The class implementations check for consistency with mir_eval,
in particular data types and shapes.
"""

import numpy as np
from dataclasses import dataclass


# Check mir_eval compliance
def check_types(attribute_names, attributes, expected_type, expected_dtype):
    """
    Check that attribute types are compliant with mir_eval in terms of data types.

    Parameters
    ----------
    attribute_names (list): list of str with attribute names for error message display.
    attributes (list): list of attributes of the class to check.
    expected_type (list): list with the expected type per attribute (usually arrays or lists).
    expected_dtype (list): list with the expected type of annotations expected (usually float or str).
    """
    for name, attribute, exp_type, int_type in zip(
        attribute_names, attributes, expected_type, expected_dtype
    ):
        if attribute is not None:
            if not isinstance(attribute, exp_type):
                raise TypeError(
                    f"{name} should be a {exp_type}, but is a {type(attribute)} instead"
                )
            if not np.asarray(attribute).size == 0:  # check list/ndarray not empty
                if all(isinstance(type(x), int_type) for x in attribute):
                    outlier = [
                        type(x)
                        for x in attribute
                        if type(x) is not int_type and type(x) is not None
                    ]
                    raise TypeError(
                        f"{name} should contain only {int_type} elements, but contains an {outlier}"
                    )


def check_lengths(attributes):
    """
    Check that attributes are compliant with mir_eval in terms of shape. In particular,
    we check if attributes have the same length.

    Parameters
    ----------
    attributes (list): list of attributes to check.


    """
    if len(attributes) > 1:
        for att1, att2 in zip(attributes[:1], attributes[1:]):
            if att1 is not None and att2 is not None and not len(att1) == len(att2):
                raise ValueError("There is a length mismatch between inputs")


def check_intervals(intervals):
    """
    Check shape of intervals to be compliant with mir_eval, that is, to be numpy
    arrays with two columns.

    Parameters
    ----------
    intervals (np.array): intervals to check for compliance.

    """
    if intervals is not None and not np.shape(intervals)[1] == 2:
        raise ValueError(
            f"Intervals should be arrays with two columns, but array has {np.shape(intervals)[1]}"
        )



# Class definitions
@dataclass
class BeatData:
    """
    mirdata BeatData class

    Usage example:
    dataset = mirdata.Dataset('beatles')  # get the beatles dataset
    track = dataset.choice_track()  # load a random track
    print(track.beats.times) >>> array([ 0.499,  1.068,  1.567, ...])
    print(track.beats.positions) >>> array([1, 2, 3, ... ])

    Attributes:
        times (np.array): time stamps of beat events in seconds.
        positions (np.array): array of integers indicating the position
                              of the beat in the metric cycle.

    """

    # declare attributes for __repr__
    times: np.ndarray
    positions: np.ndarray = None

    # customized __init__ to check data types
    def __init__(self, times, positions=None):
        attributes = [times, positions]  # list of attributes
        attribute_names = ["times", "positions"]
        expected_type = [np.ndarray, np.ndarray]
        expected_dtype = [float, int]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)

        self.times = times
        self.positions = positions


@dataclass
class SectionData:
    """
    mirdata SectionData class

    Usage example:
    dataset = mirdata.Dataset('beatles')  # get the beatles dataset
    track = dataset.choice_track()  # load a random track
    print(track.sections.intervals) >>> array([[ 0.   , 20.533], [20.533, 44.821], ... ])
    print(track.sections.labels) >>> ['intro', 'refrain', ... ]

    Attributes:
        intervals (np.array): time stamps of beginning and end of sections
                              in seconds in the shape [begin, end]
        labels (list): list of strings indicating labels of the sections (e.g. "intro", "A")
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    labels: list

    # customized __init__ to check data types
    def __init__(self, intervals, labels):
        attributes = [intervals, labels]
        attribute_names = ["intervals", "labels"]
        expected_type = [np.ndarray, list]
        expected_dtype = [float, str]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.labels = labels


@dataclass
class ChordData:
    """
    mirdata ChordData class

    Usage example:
    dataset = mirdata.Dataset('beatles')  # get the beatles dataset
    track = dataset.choice_track()  # load a random track
    print(track.chords.intervals) >>> array([[ 0.      ,  0.51721 ], [ 0.51721 ,  1.148857], ...])
    print(track.chords.labels) >>> ['N', 'G:(1)', ... ]

    Attributes:
        intervals (np.array): time stamps of beginning and end of chords
                              in seconds in the shape [begin, end].
        labels (list): list of strings indicating labels of the chords (e.g. "A", "Bb")
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    labels: list

    # customized __init__ to check data types
    def __init__(self, intervals, labels):
        attributes = [intervals, labels]
        attribute_names = ["intervals", "labels"]
        expected_type = [np.ndarray, list]
        expected_dtype = [float, str]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.labels = labels


@dataclass
class NoteData:
    """
    mirdata NoteData class

    Usage example:
    dataset = mirdata.Dataset('cante100')  # get the cante100 dataset
    track = dataset.choice_track()  # load a random track
    print(track.notes.intervals) >>> array([[ 0.281542 ,  2.568712 ], [ 4.31601  ,  4.713652 ], ... ])
    print(track.notes.frequencies) >>> array([220. , 207.65234879, 220 , 207.65234879, ...])
    print(track.notes.confidence) >>> array([1., 1., 1., 1., 1., 1, ...])

    Attributes:
        intervals (np.array): time stamps of beginning and end of notes
                              in seconds in the shape [begin, end]
        frequencies (np.array): array of frequency values in Hz
        confidence: (np. array): array of confidence values of the annotations (between 0 and 1)
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    frequencies: np.ndarray
    confidence: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, frequencies, confidence):
        attributes = [intervals, frequencies, confidence]
        attribute_names = ["intervals", "frequencies", "confidence"]
        expected_type = [np.ndarray, np.ndarray, np.ndarray]
        expected_dtype = [float, float, float]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.frequencies = frequencies
        self.confidence = confidence


@dataclass
class F0Data:
    """
    mirdata F0Data class

    Usage example:
    dataset = mirdata.Dataset('cante100')  # get the cante100 dataset
    track = dataset.choice_track()  # load a random track
    print(track.melody.times) >>> array([2.32199540e-02, 2.61224480e-02, 2.90249420e-02, ...])
    print(track.melody.frequencies) >>> array([-440., -440., -440., ...])
    print(track.melody.confidence) >>> array([0., 0., 0., ...])

    Attributes:
        intervals (np.array): time stamps of melody events in seconds
        frequencies (np.array): array of frequency values in Hz
        confidence: (np. array): array of confidence values of the annotations (between 0 and 1)
    """

    # declare attributes for __repr__
    times: np.ndarray
    frequencies: np.ndarray
    confidence: np.ndarray

    # customized __init__ to check data types
    def __init__(self, times, frequencies, confidence):
        attributes = [times, frequencies, confidence]
        attribute_names = ["times", "frequencies", "confidence"]
        expected_type = [np.ndarray, np.ndarray, np.ndarray]
        expected_dtype = [float, float, float]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)

        self.times = times
        self.frequencies = frequencies
        self.confidence = confidence


@dataclass
class MultiF0Data:
    """
    mirdata MultiF0Data class

    Usage example:
    dataset = mirdata.Dataset('medleydb_melody')  # get the medleydb_melody dataset
    track = dataset.choice_track()  # load a random track
    print(track.melody3.times) >>> array([0.04643991, 0.0522449 , 0.12190476, ...])
    print(track.melody.frequency_list) >>> [[0.0, 0.0, ...], [965.992, 996.468, ...], [987.32, 987.932, ...]]
    print(track.melody.confidence_list) >>> [[0.0, 0.0, ...], [1.0, 1.0, ...], [1.0, 1.0, ...]]

    Attributes:
        times (np.array): time stamps of melody events in seconds
        frequency_list (list): list of arrays of frequency values in Hz
        confidence: (list): list of arrays of confidence values of the annotations (between 0 and 1)
    """

    # declare attributes for __repr__
    times: np.ndarray
    frequency_list: list
    confidence_list: list

    # customized __init__ to check data types
    def __init__(self, times, frequency_list, confidence_list):
        attributes = [times, frequency_list, confidence_list]
        attribute_names = ["times", "frequency_list", "confidence_list"]
        expected_type = [np.ndarray, list, list]
        expected_dtype = [float, np.ndarray, np.ndarray]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)

        self.times = times
        self.frequency_list = frequency_list
        self.confidence_list = confidence_list


@dataclass
class KeyData:
    """
    mirdata KeyData class

    Usage example:
    dataset = mirdata.Dataset('beatles')  # get the beatles dataset
    track = dataset.choice_track()  # load a random track
    print(track.key.intervals) >>> array([[ 0.   , 44.868], [44.868, 78.893]])
    print(track.key.labels) >>> ['F', 'G']

    Attributes:
        intervals (np.array): time stamps of beginning and end of key sections
                              in seconds in the shape [begin, end].
        labels (list): list of strings indicating the key (e.g. "F", "G")

    """

    # declare attributes for __repr__
    intervals: np.ndarray
    labels: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, labels):
        attributes = [intervals, labels]
        expected_type = [np.ndarray, list]
        attribute_names = ["intervals", "labels"]
        expected_dtype = [float, str]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.labels = labels


@dataclass
class LyricData:
    """
    mirdata LyricData class

    Usage example:
    dataset = mirdata.Dataset('ikala')  # get the ikala dataset
    track = dataset.choice_track()  # load a random track
    print(track.lyrics.intervals) >>> array([[0.027, 0.232], [0.232, 0.968], ...])
    print(track.lyrics.text) >>> ['JUST', 'WANNA', ...]
    print(track.lyrics.pronunciations) >>> [None, None, ...]

    Attributes:
        intervals (np.array): time stamps of beginning and end of lyric sections
                              in seconds in the shape [begin, end].
        text (list): list of strings with the text of the lyrics
        pronunciations (list): list of strings with the pronunciation of the lyrics


    """

    # declare attributes for __repr__
    intervals: np.ndarray
    text: np.ndarray
    pronunciations: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, text, pronunciations):
        attributes = [intervals, text, pronunciations]
        attribute_names = ["intervals", "text", "pronunciations"]
        expected_type = [np.ndarray, list, list]
        expected_dtype = [float, str, str]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.text = text
        self.pronunciations = pronunciations


@dataclass
class TempoData:
    """
    mirdata TempoData class

    Usage example:
    dataset = mirdata.Dataset('giantsteps_tempo')  # get the giantsteps_tempo dataset
    track = dataset.choice_track()  # load a random track
    print(track.tempo.intervals) >>> array([[  0., 120.], ... ])
    print(track.tempo.value) >>> array([137.6,  ...])
    print(track.tempo.confidence) >>> array([1., ...])

    Attributes:
        intervals (np.array): time stamps of beginning and end of local tempo section
                              in seconds in the shape [begin, end]
        value (np.array): tempo value in bpm
        confidence: (np. array): array of confidence values of the annotations (between 0 and 1)
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    value: np.ndarray
    confidence: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, value, confidence):
        attributes = [intervals, value, confidence]
        attribute_names = ["intervals", "value", "confidence"]
        expected_type = [np.ndarray, np.ndarray, np.ndarray]
        expected_dtype = [float, float, float]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.value = value
        self.confidence = confidence


@dataclass
class EventData:
    """
    mirdata EventData class

    Usage example:
    dataset = mirdata.Dataset('rwc_popular')  # get the rwc_popular dataset
    track = dataset.choice_track()  # load a random track
    print(track.vocal_instrument_activity.intervals) >>> array([[ 0., 10.29306122],[10.29306122, 11.88349206], ...])
    print(track.vocal_instrument_activity.event) >>> [['b', 'm:withm', 'b', 'm:withm', ...]

    Attributes:
        intervals (np.array): time stamps of beginning and end of events
                              in seconds in the shape [begin, end].
        event (list): list of strings indicating the event


    """

    # declare attributes for __repr__
    intervals: np.ndarray
    event: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, event):
        attributes = [intervals, event]
        expected_type = [np.ndarray, list]
        attribute_names = ["intervals", "event"]
        expected_dtype = [float, str]

        # check mir_eval compliance
        check_types(attribute_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.event = event
