# -*- coding: utf-8 -*-
"""
    TODO all docs
    """

import numpy as np
from dataclasses import dataclass


def check_types(attributes_names, attributes, expected_type, expected_dtype):
    for name, attribute, exp_type, int_type in zip(attributes_names, attributes, expected_type, expected_dtype):
        if attribute is not None:
            if not isinstance(attribute, exp_type):
                raise TypeError(f"{name} should be a {exp_type}, but is a {type(attribute)} instead")
            if all(isinstance(type(x), int_type) for x in attribute):
                raise TypeError(f"{name} should contain only {int_type} elements")


def check_lengths(attributes):
    if len(attributes)>1:
        for att1, att2 in zip(attributes[:1], attributes[1:]):
            if att1 is not None and att2 is not None and not len(att1) == len(att2):
                raise ValueError("There is a length mismatch between inputs")


def check_intervals(intervals):
    if intervals is not None and not np.shape(intervals)[1] == 2:
            raise ValueError(f"Intervals should be arrays with two columns, but array has {np.shape(intervals)[1]}")


@dataclass
class BeatData:
    """
    # BeatData = namedtuple("BeatData", ["beat_times", "beat_positions"])
    """
    # declare attributes for __repr__
    beat_times: np.ndarray
    beat_positions: np.ndarray = None

    # customized __init__ to check data types
    def __init__(self, beat_times, beat_positions=None):
        attributes = [beat_times, beat_positions]  # list of attributes
        attributes_names = ["beat_times", "beat_positions"]
        expected_type = [np.ndarray, np.ndarray]
        expected_dtype = [float, int]

        # check mir_eval complience
        check_types(attributes_names, attributes, expected_type, expected_dtype)
        check_lengths(attributes)

        self.beat_times = beat_times
        self.beat_positions = beat_positions


@dataclass
class SectionData:
    """
    # SectionData = namedtuple("SectionData", ["intervals", "labels"])
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



# NoteData = namedtuple("NoteData", ["intervals", "notes", "confidence"])
#
# F0Data = namedtuple("F0Data", ["times", "frequencies", "confidence"])
#
# MultipitchData = namedtuple(
#     "MultipitchData", ["times", "frequency_list", "confidence_list"]
# )
#
# LyricData = namedtuple(
#     "LyricData", ["start_times", "end_times", "lyrics", "pronunciations"]
# )
#

#
#
#
# ChordData = namedtuple("ChordData", ["intervals", "labels"])
#
# KeyData = namedtuple("KeyData", ["start_times", "end_times", "keys"])
#
# TempoData = namedtuple("TempoData", ["time", "duration", "value", "confidence"])
#
# EventData = namedtuple("EventData", ["start_times", "end_times", "event"])
#
