"""mirdata annotation data types
"""
import numpy as np


class Annotation(object):
    """Annotation base class"""

    def __repr__(self):
        attributes = [v for v in dir(self) if not v.startswith("_")]
        repr_str = f"{self.__class__.__name__}({', '.join(attributes)})"
        return repr_str


class BeatData(Annotation):
    """BeatData class

    Attributes:
        times (np.ndarray): array of time stamps (as floats) in seconds
            with positive, strictly increasing values
        positions (np.ndarray or None): array of beat positions (as ints)
            e.g. 1, 2, 3, 4

    """

    def __init__(self, times, positions=None):
        validate_array_like(times, np.ndarray, float)
        validate_array_like(positions, np.ndarray, int, none_allowed=True)
        validate_lengths_equal([times, positions])
        validate_times(times)

        self.times = times
        self.positions = positions


class SectionData(Annotation):
    """SectionData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            times should be positive and intervals should have
            non-negative duration
        labels (list or None): list of labels (as strings)

    """

    def __init__(self, intervals, labels=None):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str, none_allowed=True)
        validate_lengths_equal([intervals, labels])
        validate_intervals(intervals)

        self.intervals = intervals
        self.labels = labels


class NoteData(Annotation):
    """NoteData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        notes (np.ndarray): array of notes (as floats) in Hz
        confidence (np.ndarray or None): array of confidence values
            between 0 and 1

    """

    def __init__(self, intervals, notes, confidence=None):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(notes, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, notes, confidence])
        validate_intervals(intervals)
        validate_confidence(confidence)

        self.intervals = intervals
        self.notes = notes
        self.confidence = confidence


class ChordData(Annotation):
    """ChordData class

    Attributes:
        intervals (np.ndarray or None): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        labels (list): list chord labels (as strings)
        confidence (np.ndarray or None): array of confidence values
            between 0 and 1

    """

    def __init__(self, intervals, labels, confidence=None):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, labels, confidence])
        validate_intervals(intervals)
        validate_confidence(confidence)

        self.intervals = intervals
        self.labels = labels
        self.confidence = confidence


class F0Data(Annotation):
    """F0Data class

    Attributes:
        times (np.ndarray): array of time stamps (as floats) in seconds
            with positive, strictly increasing values
        frequencies (np.ndarray): array of frequency values (as floats)
            in Hz
        confidence (np.ndarray or None): array of confidence values
            between 0 and 1

    """

    def __init__(self, times, frequencies, confidence=None):
        validate_array_like(times, np.ndarray, float)
        validate_array_like(frequencies, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([times, frequencies, confidence])
        validate_times(times)
        validate_confidence(confidence)

        self.times = times
        self.frequencies = frequencies
        self.confidence = confidence


class MultiF0Data(Annotation):
    """MultiF0Data class

    Attributes:
        times (np.ndarray): array of time stamps (as floats) in seconds
            with positive, strictly increasing values
        frequency_list (list): list of lists of frequency values (as floats)
            in Hz
        confidence_list (list or None): list of lists of confidence values
            between 0 and 1

    """

    def __init__(self, times, frequency_list, confidence_list=None):
        validate_array_like(times, np.ndarray, float)
        validate_array_like(frequency_list, list, list)
        validate_array_like(confidence_list, list, list, none_allowed=True)
        validate_lengths_equal([times, frequency_list, confidence_list])
        validate_times(times)

        self.times = times
        self.frequency_list = frequency_list
        self.confidence_list = confidence_list


class KeyData(Annotation):
    """KeyData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        keys (list): list key labels (as strings)

    """

    def __init__(self, intervals, keys):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(keys, list, str)
        validate_lengths_equal([intervals, keys])
        validate_intervals(intervals)

        self.intervals = intervals
        self.keys = keys


class LyricData(Annotation):
    """LyricData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        lyrics (list): list of lyrics (as strings)
        pronunciations (list or None): list of pronunciations (as strings)

    """

    def __init__(self, intervals, lyrics, pronunciations=None):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(lyrics, list, str)
        validate_array_like(pronunciations, list, str, none_allowed=True)
        validate_lengths_equal([intervals, lyrics, pronunciations])
        validate_intervals(intervals)

        self.intervals = intervals
        self.lyrics = lyrics
        self.pronunciations = pronunciations


class TempoData(Annotation):
    """TempoData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        value (list): array of tempo values (as floats)
        confidence (np.ndarray or None): array of confidence values
            between 0 and 1

    """

    def __init__(self, intervals, value, confidence=None):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(value, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, value, confidence])
        validate_intervals(intervals)
        validate_confidence(confidence)

        self.intervals = intervals
        self.value = value
        self.confidence = confidence


class EventData(Annotation):
    """TempoData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        events (list): list of event labels (as strings)

    """

    def __init__(self, intervals, events):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(events, list, str)
        validate_lengths_equal([intervals, events])
        validate_intervals(intervals)

        self.intervals = intervals
        self.events = events


def validate_array_like(array_like, expected_type, expected_dtype, none_allowed=False):
    """Validate that array-like object is well formed

    If array_like is None, validation passes automatically.

    Args:
        array_like (array-like): object to validate
        expected_type (type): expected type, either list or np.ndarray
        expected_dtype (type): expected dtype
        none_allowed (bool): if True, allows array to be None

    Raises:
        TypeError: if type/dtype does not match expected_type/expected_dtype
        ValueError: if array

    """
    if array_like is None:
        if none_allowed:
            return
        else:
            raise ValueError("array_like cannot be None")

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
    """Validate that arrays in list are equal in length

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


def validate_confidence(confidence):
    """Validate if confidence is well-formed.

    If confidence is None, validation passes automatically

    Args:
        confidence (np.ndarray): an array of confidence values

    Raises:
        ValueError: if confidence are not between 0 and 1

    """
    if confidence is None:
        return

    confidence_shape = np.shape(confidence)
    if len(confidence_shape) != 1:
        raise ValueError(
            f"Confidence should be 1d, but array has shape {confidence_shape}"
        )

    if (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("confidence should be between 0 and 1")


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

    time_shape = np.shape(times)
    if len(time_shape) != 1:
        raise ValueError(f"Times should be 1d, but array has shape {time_shape}")

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
    interval_shape = np.shape(intervals)
    if len(interval_shape) != 2 or interval_shape[1] != 2:
        raise ValueError(
            f"Intervals should be arrays with two columns, but array has {interval_shape}"
        )

    # validate that time stamps are all positive numbers
    if (intervals < 0).any():
        raise ValueError(f"Interval values should be nonnegative numbers")

    # validate that end times are bigger than start times
    elif (intervals[:, 1] - intervals[:, 0] < 0).any():
        raise ValueError(f"Interval start times must be smaller than end times")
