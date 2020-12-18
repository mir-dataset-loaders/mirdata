# -*- coding: utf-8 -*-
"""mirdata annotation data types
"""

import numpy as np


class BeatData:
    """
    """

    def __init__(self, times, positions=None):
        # check mir_eval compliance
        validate_array_like(times, np.ndarray, float)
        validate_array_like(positions, np.ndarray, int)
        validate_lengths_equal([times, positions])

        self.times = times
        self.positions = positions

    def __repr__(self):
        pass


class SectionData:
    """
    """

    def __init__(self, intervals, labels):
        # check mir_eval compliance
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str)
        validate_lengths_equal([intervals, labels])
        validate_intervals(intervals)

        self.intervals = intervals
        self.labels = labels


class NoteData:
    """

    """

    # customized __init__ to check data types
    def __init__(self, intervals, notes, confidence=None):

        # check mir_eval compliance
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(notes, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float)
        validate_lengths_equal([intervals, notes, confidence])
        validate_intervals(intervals)

        self.intervals = intervals
        self.notes = notes
        self.confidence = confidence


class ChordData:
    """
    # ChordData = namedtuple("ChordData", ["intervals", "labels"])
    """

    def __init__(self, intervals, labels, confidence=None):

        # check mir_eval compliance
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float)
        validate_lengths_equal([intervals, labels, confidence])
        validate_intervals(intervals)

        self.intervals = intervals
        self.labels = labels
        self.confidence = confidence


class F0Data:
    """
    # F0Data = namedtuple("F0Data", ["times", "frequencies", "confidence"])
    """

    # customized __init__ to check data types
    def __init__(self, times, frequencies, confidence=None):

        # check mir_eval compliance
        validate_array_like(times, np.ndarray, float)
        validate_array_like(frequencies, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float)
        validate_lengths_equal([times, frequencies, confidence])

        self.times = times
        self.frequencies = frequencies
        self.confidence = confidence


class MultiF0Data:
    """
    # MultipitchData = namedtuple(
#     "MultipitchData", ["times", "frequency_list", "confidence_list"]
# )
    """

    # customized __init__ to check data types
    def __init__(self, times, frequency_list, confidence_list=None):
        # check mir_eval compliance
        validate_array_like(times, np.ndarray, float)
        validate_array_like(frequency_list, list, list)
        validate_array_like(confidence_list, list, list)
        validate_lengths_equal([times, frequency_list, confidence_list])

        self.times = times
        self.frequency_list = frequency_list
        self.confidence_list = confidence_list


class KeyData:
    """
    # KeyData = namedtuple("KeyData", ["start_times", "end_times", "keys"])
    """

    # customized __init__ to check data types
    def __init__(self, intervals, keys):

        # check mir_eval compliance
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(keys, list, str)
        validate_lengths_equal([intervals, keys])
        validate_intervals(intervals)

        self.intervals = intervals
        self.keys = keys


class LyricData:
    """
    # LyricData = namedtuple(
    #     "LyricData", ["start_times", "end_times", "lyrics", "pronunciations"]
    # )
    """

    # customized __init__ to check data types
    def __init__(self, intervals, lyrics, pronunciations=None):

        # check mir_eval compliance
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(lyrics, list, str)
        validate_array_like(pronunciations, list, str)
        validate_lengths_equal([intervals, lyrics, pronunciations])
        validate_intervals(intervals)

        self.intervals = intervals
        self.lyrics = lyrics
        self.pronunciations = pronunciations


class TempoData:
    """
    # TempoData = namedtuple("TempoData", ["intervals, "value", "confidence"])
    """

    # customized __init__ to check data types
    def __init__(self, intervals, value, confidence=None):

        # check mir_eval compliance
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(value, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float)
        validate_lengths_equal([intervals, value, confidence])
        validate_intervals(intervals)

        self.intervals = intervals
        self.value = value
        self.confidence = confidence


class EventData:
    """
    #
    #
    # EventData = namedtuple("EventData", ["start_times", "end_times", "event"])
    #
    """

    def __init__(self, intervals, events):

        # check mir_eval compliance
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(events, list, str)
        validate_lengths_equal([intervals, events])
        validate_intervals(intervals)

        self.intervals = intervals
        self.events = events


def validate_array_like(array_like, expected_type, expected_dtype):
    """Validate that array-like object is well formed
    
    If array_like is None, validation passes automatically.

    Args:
        array_like (array-like): object to validate
        expected_type : expected type, either list or np.ndarray
        expected_dtype : expected dtype
    
    Raises:
        TypeError: if type/dtype does not match expected_type/expected_dtype
        ValueError: if array

    """
    if array_like is None:
        return

    assert expected_type in [
        list,
        np.ndarray,
    ], "expected type must be a list or np.ndarray"

    if not isinstance(array_like, expected_type):
        raise TypeError(
            f"Object should be a {expected_type}, but is a {type(array_like)}"
        )

    if expected_type == list and not all(
        isinstance(n, expected_dtype) for n in array_like
    ):
        raise TypeError(f"List elements should all have type {expected_dtype}")

    if expected_type == np.ndarray and array_like.dtype != expected_dtype:
        raise TypeError(
            f"Array should have dtype {expected_dtype} but has {array_like.dtype}"
        )

    if np.asarray(array_like).size == 0:
        raise ValueError("Object should not be empty, use None instead")


def validate_lengths_equal(array_list):
    """Validate that arays in list are are equal in length

    Some arrays may be None, and the validation for these are skipped.

    Args:
        array_list (list): list of array-like objects
    
    Raises:
        ValueError: if arrays are not equal in length

    """
    if len(array_list) == 1:
        return

    for att1, att2 in zip(array_list[:1], array_list[1:]):
        if att1 is None or att2 is None:
            continue

        if not len(att1) == len(att2):
            raise ValueError("Arrays have unequal length")


def validate_times(times):
    """Validate if times are well-formed.

    If times is None, validation passes automatically

    Args:
        times (np.ndarray): an array of time stamps

    Raises:
        ValueError: if times have negative values or are non-increasing

    """
    if times is None:
        return

    if (times < 0).any():
        raise ValueError("times should be positive numbers")

    if (times[1:] - times[:-1] <= 0).any():
        raise ValueError("times should be strictly increasing")


def validate_intervals(intervals):
    """Validate if intervals are well-formed.

    If intervals is None, validation passes automatically

    Args:
        intervals (np.ndarray): (n x 2) array

    Raises:
        ValueError: if intervals have an invalid shape, have negative values
        or if end times are smaller than start times.

    """
    if intervals is None:
        return

    # validate that intervals have the correct shape
    if not np.shape(intervals)[1] == 2:
        raise ValueError(
            f"Intervals should be arrays with two columns, but array has {np.shape(intervals)[1]}"
        )

    # validate that time stamps are all positive numbers
    if (intervals < 0).any():
        raise ValueError(f"Interval values should be nonnegative numbers")

    # validate that end times are bigger than start times
    elif (intervals[:, 1] - intervals[:, 0] < 0).any():
        raise ValueError(f"Interval start times must be smaller than end times")
