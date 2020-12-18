# -*- coding: utf-8 -*-
"""
    TODO all docs
    """

import numpy as np
from dataclasses import dataclass


# Check consistency


def check_types(attributes_names, attributes, expected_type, expected_dtype):
    for name, attribute, exp_type, int_type in zip(
        attributes_names, attributes, expected_type, expected_dtype
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
    if len(attributes) > 1:
        for att1, att2 in zip(attributes[:1], attributes[1:]):
            if att1 is not None and att2 is not None and not len(att1) == len(att2):
                raise ValueError("There is a length mismatch between inputs")


def check_intervals(intervals):
    if intervals is not None and not np.shape(intervals)[1] == 2:
        raise ValueError(
            f"Intervals should be arrays with two columns, but array has {np.shape(intervals)[1]}"
        )


# Class definitions


@dataclass
class BeatData:
    """
    """

    # declare attributes for __repr__
    times: np.ndarray
    positions: np.ndarray = None

    # customized __init__ to check data types
    def __init__(self, times, positions=None):
        attributes = [times, positions]  # list of attributes
        attributes_names = ["times", "positions"]
        expected_type = [np.ndarray, np.ndarray]
        expected_dtype = [float, int]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)

        self.times = times
        self.positions = positions


@dataclass
class SectionData:
    """
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    labels: list

    # customized __init__ to check data types
    def __init__(self, intervals, labels):
        attributes = [intervals, labels]
        attributes_names = ["intervals", "labels"]
        expected_type = [np.ndarray, list]
        expected_dtype = [float, str]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.labels = labels


@dataclass
class NoteData:
    """

    """

    # declare attributes for __repr__
    intervals: np.ndarray
    notes: np.ndarray
    confidence: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, notes, confidence):
        attributes = [intervals, notes, confidence]
        attributes_names = ["intervals", "notes", "confidence"]
        expected_type = [np.ndarray, np.ndarray, np.ndarray]
        expected_dtype = [float, float, float]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.notes = notes
        self.confidence = confidence


@dataclass
class ChordData:
    """
    # ChordData = namedtuple("ChordData", ["intervals", "labels"])
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    labels: list

    # customized __init__ to check data types
    def __init__(self, intervals, labels):
        attributes = [intervals, labels]
        attributes_names = ["intervals", "labels"]
        expected_type = [np.ndarray, list]
        expected_dtype = [float, str]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.labels = labels


@dataclass
class F0Data:
    """
    # F0Data = namedtuple("F0Data", ["times", "frequencies", "confidence"])
    """

    # declare attributes for __repr__
    times: np.ndarray
    frequencies: np.ndarray
    confidence: np.ndarray

    # customized __init__ to check data types
    def __init__(self, times, frequencies, confidence):
        attributes = [times, frequencies, confidence]
        attributes_names = ["times", "frequencies", "confidence"]
        expected_type = [np.ndarray, np.ndarray, np.ndarray]
        expected_dtype = [float, float, float]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)

        self.times = times
        self.frequencies = frequencies
        self.confidence = confidence


@dataclass
class MultiF0Data:
    """
    # MultipitchData = namedtuple(
#     "MultipitchData", ["times", "frequency_list", "confidence_list"]
# )
    """

    # declare attributes for __repr__
    times: np.ndarray
    frequency_list: list
    confidence_list: list

    # customized __init__ to check data types
    def __init__(self, times, frequency_list, confidence_list):
        attributes = [times, frequency_list, confidence_list]
        attributes_names = ["times", "frequency_list", "confidence_list"]
        expected_type = [np.ndarray, list, list]
        expected_dtype = [float, np.ndarray, np.ndarray]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)

        self.times = times
        self.frequency_list = frequency_list
        self.confidence_list = confidence_list


@dataclass
class KeyData:
    """
    # KeyData = namedtuple("KeyData", ["start_times", "end_times", "keys"])
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    keys: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, keys):
        attributes = [intervals, keys]
        expected_type = [np.ndarray, list]
        attributes_names = ["intervals", "keys"]
        expected_dtype = [float, str]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.keys = keys


@dataclass
class LyricData:
    """
    # LyricData = namedtuple(
    #     "LyricData", ["start_times", "end_times", "lyrics", "pronunciations"]
    # )
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    lyrics: np.ndarray
    pronunciations: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, lyrics, pronunciations):
        attributes = [intervals, lyrics, pronunciations]
        attributes_names = ["intervals", "lyrics", "pronunciations"]
        expected_type = [np.ndarray, list, list]
        expected_dtype = [float, str, str]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.lyrics = lyrics
        self.pronunciations = pronunciations


@dataclass
class TempoData:
    """
    # TempoData = namedtuple("TempoData", ["intervals, "value", "confidence"])
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    value: np.ndarray
    confidence: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, value, confidence):
        attributes = [intervals, value, confidence]
        attributes_names = ["intervals", "value", "confidence"]
        expected_type = [np.ndarray, np.ndarray, np.ndarray]
        expected_dtype = [float, float, float]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.value = value
        self.confidence = confidence


@dataclass
class EventData:
    """
    #
    #
    # EventData = namedtuple("EventData", ["start_times", "end_times", "event"])
    #
    """

    # declare attributes for __repr__
    intervals: np.ndarray
    event: np.ndarray

    # customized __init__ to check data types
    def __init__(self, intervals, event):
        attributes = [intervals, event]
        expected_type = [np.ndarray, list]
        attributes_names = ["intervals", "event"]
        expected_dtype = [float, str]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)
        check_intervals(intervals)

        self.intervals = intervals
        self.event = event
