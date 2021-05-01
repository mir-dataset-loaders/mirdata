"""mirdata annotation data types
"""
import logging
import re

from jams.schema import namespace
import librosa
import numpy as np

#: Beat position units
BEAT_POSITION_UNITS = {
    "bar_index": "beat index within a bar, 1-indexed",
    "global_index": "beat index within full track, 1-indexed",
    "bar_fraction": "beat position as fractions of bars, e.g. 0.25",
    "global_fraction": "bar_frac, but where the integer part indicates the bar. e.g. 4.25",
}

#: Chord units
CHORD_UNITS = {
    "harte": "chords in harte format, e.g. Ab:maj7",
    "jams": "chords in jams 'chord' format",
    "open": "no strict schema or units",
}

#: Confidence units
CONFIDENCE_UNITS = {
    "likelihood": "score between 0 and 1",
    "velocity": "MIDI velocity between 0 and 127",
    "binary": "0 or 1",
    "energy": "energy value, measured as the sum of a squared signal",
}

#: Event units
EVENT_UNITS = {"open": "no scrict schema or units"}

#: Key units
KEY_UNITS = {"key_mode": "key labels in key-mode format, e.g. G#:minor"}

#: Lyric units
LYRIC_UNITS = {
    "words": "lyrics as words or phrases",
    "syllable_open": "lyrics segmented by syllable, no strict schema",
    "pronunciations_open": "lyric pronunciations, no strict schema",
}

#: Pitch units
PITCH_UNITS = {
    "hz": "hertz",
    "midi": "MIDI note number",
    "pc": "pitch class, e.g. G#",
    "note_name": "pc with octave, e.g. Ab4",
}

#: Section units
SECTION_UNITS = {"open": "no scrict schema or units"}

#: Tempo units
TEMPO_UNITS = {"bpm": "beats per minute"}

#: Time units
TIME_UNITS = {
    "s": "seconds",
    "ms": "miliseconds",
    "ticks": "MIDI ticks",
}

#: Voicing units
VOICING_UNITS = {
    "binary": "voicing indicators as 0 or 1",
    "continuous": "voicing indicators as continuous values between 0 and 1",
}


class Annotation(object):
    """Annotation base class"""

    def __repr__(self):
        attributes = [v for v in dir(self) if not v.startswith("_")]
        repr_str = f"{self.__class__.__name__}({', '.join(attributes)})"
        return repr_str


class BeatData(Annotation):
    """BeatData class

    Attributes:
        times (np.ndarray): array of time stamps with positive,
            strictly increasing values
        time_unit (str): time unit, one of TIME_UNITS
        positions (np.ndarray): array of beat positions in the format
            of position_unit. For all units, values of 0 indicate beats which
            fall outside of a measure.
        position_unit (str): beat position unit, one of BEAT_POSITION_UNITS
        confidence (np.ndarray): array of confidence values
        confidence_unit (str): confidence unit, one of CONFIDENCE_UNITS

    """

    def __init__(
        self,
        times,
        time_unit,
        positions,
        position_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(times, np.ndarray, float)
        validate_lengths_equal([times, positions])
        validate_times(times, time_unit)
        validate_beat_positions(positions, position_unit)
        validate_confidence(confidence, confidence_unit)

        self.times = times
        self.time_unit = time_unit
        self.positions = positions
        self.position_unit = position_unit
        self.confidence = confidence
        self.confidence_unit = confidence_unit


class SectionData(Annotation):
    """SectionData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        labels (list or None): list of section labels
        label_unit (str or None): label unit, one of SECTION_UNITS

    """

    def __init__(self, intervals, interval_unit, labels=None, label_unit=None):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str, none_allowed=True)
        validate_lengths_equal([intervals, labels])
        validate_intervals(intervals, interval_unit)
        validate_unit(label_unit, SECTION_UNITS, allow_none=True)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.labels = labels
        self.label_unit = label_unit


class NoteData(Annotation):
    """NoteData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        pitches (np.ndarray): array of pitches
        pitch_unit (str): note unit, one of PITCH_UNITS
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of CONFIDENCE_UNITS

    """

    def __init__(
        self,
        intervals,
        interval_unit,
        pitches,
        pitch_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(pitches, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, pitches, confidence])
        validate_intervals(intervals, interval_unit)
        validate_pitches(pitches, pitch_unit)
        validate_confidence(confidence, confidence_unit)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.pitches = pitches
        self.pitch_unit = pitch_unit
        self.confidence = confidence
        self.confidence_unit = confidence_unit

    @property
    def notes(self):
        logging.warning(
            "Deprecation warning: NoteData.notes will be removed in a future version."
            + "Use NoteData.pitches"
        )
        return self.pitches


class ChordData(Annotation):
    """ChordData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        labels (list): list chord labels (as strings)
        label_unit (str): chord label schema
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of CONFIDENCE_UNITS
    """

    def __init__(
        self,
        intervals,
        interval_unit,
        labels,
        label_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, labels, confidence])
        validate_intervals(intervals, interval_unit)
        validate_unit(label_unit, CHORD_UNITS)
        validate_chord_labels(labels, label_unit)
        validate_confidence(confidence, confidence_unit)

        self.intervals = intervals
        self.labels = labels
        self.confidence = confidence


class F0Data(Annotation):
    """F0Data class

    Attributes:
        times (np.ndarray): array of time stamps (as floats)
            with positive, strictly increasing values
        time_unit (str): time unit, one of TIME_UNITS
        frequencies (np.ndarray): array of frequency values (as floats)
        frequency_unit (str): frequency unit, one of PITCH_UNITS
        voicing (np.ndarray): array of voicing values, indicating whether or
            not a time frame has an active pitch
        voicing_unit (str): voicing unit, one of VOICING_UNITS
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of CONFIDENCE_UNITS

    """

    def __init__(
        self,
        times,
        time_unit,
        frequencies,
        frequency_unit,
        voicing,
        voicing_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(times, np.ndarray, float)
        validate_array_like(frequencies, np.ndarray, float)
        validate_array_like(voicing, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([times, frequencies, voicing, confidence])
        validate_times(times, time_unit)
        validate_pitches(frequencies, frequency_unit)
        validate_voicing(voicing, voicing_unit)
        validate_confidence(confidence, confidence_unit)
        if any(voicing[frequencies == 0] != 0):
            raise ValueError("Found frequencies with value 0, but a nonzero voicing.")

        self.times = times
        self.time_unit = time_unit
        self.frequencies = frequencies
        self.frequency_unit = frequency_unit
        self.voicing = voicing
        self.voicing_unit = voicing_unit
        self._confidence = confidence
        self.confidence_unit = confidence_unit

    @property
    def confidence(self):
        logging.warning(
            "Warning: the AIP for annotations.F0Data.confidence has changed. "
            + "For most datasets, confidence will now be None, and "
            + "F0Data.voicing should be used instead."
        )
        return self._confidence


class MultiF0Data(Annotation):
    """MultiF0Data class

    Attributes:
        times (np.ndarray): array of time stamps (as floats)
            with positive, strictly increasing values
        time_unit (str): time unit, one of TIME_UNITS
        frequency_list (list): list of lists of frequency values (as floats)
        frequency_unit (str): frequency unit, one of PITCH_UNITS
        confidence_list (np.ndarray or None): list of lists of confidence values
        confidence_unit (str or None): confidence unit, one of CONFIDENCE_UNITS

    """

    def __init__(
        self,
        times,
        time_unit,
        frequency_list,
        frequency_unit,
        confidence_list=None,
        confidence_unit=None,
    ):
        validate_array_like(times, np.ndarray, float)
        validate_array_like(frequency_list, list, list)
        validate_array_like(confidence_list, list, list, none_allowed=True)
        validate_lengths_equal([times, frequency_list, confidence_list])
        validate_times(times, time_unit)
        validate_pitches(frequency_list, frequency_unit)
        validate_confidence(confidence_list, confidence_unit)

        self.times = times
        self.time_unit = time_unit
        self.frequency_list = frequency_list
        self.frequency_unit = frequency_unit
        self.confidence_list = confidence_list
        self.confidence_unit = confidence_unit


class KeyData(Annotation):
    """KeyData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        keys (list): list key labels (as strings)
        key_unit (str): key unit, one of KEY_UNITS

    """

    def __init__(self, intervals, interval_unit, keys, key_unit):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(keys, list, str)
        validate_lengths_equal([intervals, keys])
        validate_intervals(intervals, interval_unit)
        validate_key_labels(keys, key_unit)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.keys = keys
        self.key_unit = key_unit


class LyricData(Annotation):
    """LyricData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        lyrics (list): list of lyrics (as strings)
        lyric_unit (str): lyric unit, one of LYRIC_UNITS

    """

    def __init__(
        self,
        intervals,
        interval_unit,
        lyrics,
        lyric_unit,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(lyrics, list, str)
        validate_lengths_equal([intervals, lyrics])
        validate_intervals(intervals, interval_unit)
        validate_unit(lyric_unit, LYRIC_UNITS)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.lyrics = lyrics
        self.lyric_unit = lyric_unit

    @property
    def pronunciations(self):
        logging.warning(
            "LyricData.pronunciations will be removed in a future version. "
            + "Use LyricData.lyrics"
        )
        return self.lyrics


class TempoData(Annotation):
    """TempoData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        tempos (list): array of tempo values (as floats)
        tempo_unit (str): tempo unit, one of TEMPO_UNITS
        confidence (np.ndarray or None): array of confidence values
        confidence_unit (str or None): confidence unit, one of CONFIDENCE_UNITS

    """

    def __init__(
        self,
        intervals,
        interval_unit,
        tempos,
        tempo_unit,
        confidence=None,
        confidence_unit=None,
    ):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(tempos, np.ndarray, float)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, tempos, confidence])
        validate_intervals(intervals, interval_unit)
        validate_tempos(tempos, tempo_unit)
        validate_confidence(confidence, confidence_unit)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.tempos = tempos
        self.tempo_unit = tempo_unit
        self.confidence = confidence
        self.confidence_unit = confidence_unit

    @property
    def value(self):
        logging.warning(
            "Deprecation warning: TempoData.value will be removed in future versions. Use TempoData.tempos instead."
        )
        return self.tempos


class EventData(Annotation):
    """EventData class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            in the form [start_time, end_time]. Times should be positive
            and intervals should have non-negative duration
        interval_unit (str): unit of the time values in intervals. One
            of TIME_UNITS.
        interval_unit (str): interval units, one of TIME_UNITS
        events (list): list of event labels (as strings)
        event_unit (str): event units, one of EVENT_UNITS

    """

    def __init__(self, intervals, interval_unit, events, event_unit):
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(events, list, str)
        validate_lengths_equal([intervals, events])
        validate_intervals(intervals, interval_unit)
        validate_unit(event_unit, EVENT_UNITS)

        self.intervals = intervals
        self.interval_unit = interval_unit
        self.events = events
        self.event_unit = event_unit


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


def validate_tempos(tempo, tempo_unit):
    """Validate if tempos are well-formed

    Args:
        tempo (list): list of tempo values
        tempo_unit (str): tempo unit, one of TEMPO_UNITS

    Raises:
        ValueError: if tempos are not well-formed
    """
    validate_unit(tempo_unit, TEMPO_UNITS)
    if (tempo < 0).any():
        raise ValueError("tempos must be positive")


def validate_beat_positions(positions, position_unit):
    """Validate if positions is well-formed.

    Args:
        positions (np.ndarray): an array of positions values
        positions_unit (str): one of BEAT_POSITION_UNITS

    Raises:
        ValueError: if positions values are incompatible with the unit

    """
    validate_unit(position_unit, BEAT_POSITION_UNITS)

    position_shape = np.shape(positions)
    if len(position_shape) != 1:
        raise ValueError(
            f"positions should be 1d, but array has shape {position_shape}"
        )

    if (positions < 0).any():
        raise ValueError("beat positions must be positive. Found values below 0.")

    if position_unit in ["bar_index", "global_index"] and not np.array_equal(
        np.floor(positions), positions
    ):
        raise ValueError(
            "measure index or global indexes should be integers. "
            + "Found fractional values."
        )

    # we expect no more than 32 beats per bar - this can be changed if a need arises!
    if position_unit == "bar_index" and np.max(positions) > 32:
        raise ValueError(
            "beats with bar_index units should have indexes "
            + "which start from 1 at the beginning of every measure. "
            + "Found values > 16."
        )

    if position_unit == "bar_fraction" and np.max(positions) > 1:
        raise ValueError(
            "beats with bar_fraction units should be between 0 and 1. "
            + "Found values above 1."
        )


def validate_confidence(confidence, confidence_unit):
    """Validate if confidence is well-formed.

    If confidence is None, validation passes automatically

    Args:
        confidence (np.ndarray): an array of confidence values
        confidence_unit (str): one of CONFIDENCE_UNITS

    Raises:
        ValueError: if confidence values are incompatible with the unit

    """
    if confidence is None:
        return

    validate_unit(confidence_unit, CONFIDENCE_UNITS)
    if isinstance(confidence[0], list):
        confidence_flat = [c for subconf in confidence for c in subconf]
    else:
        confidence_flat = confidence

    if confidence_unit == "likelihood" and (
        any([c < 0 for c in confidence_flat]) or any([c > 1 for c in confidence_flat])
    ):
        raise ValueError(
            "confidence with unit 'likelihood' should be between 0 and 1. "
            + "Found values outside [0, 1]."
        )

    if confidence_unit == "energy" and any([c < 0 for c in confidence_flat]):
        raise ValueError(
            "confidence with unit 'energy' should be nonnegative. "
            + "Found negative values."
        )

    if confidence_unit == "binary" and any([c not in [0, 1] for c in confidence_flat]):
        raise ValueError(
            "confidence with unit 'binary' should only have values of 0 or 1. "
            + "Found non-binary values."
        )

    if confidence_unit == "velocity" and (
        any([c < 0 for c in confidence_flat]) or any([c > 127 for c in confidence_flat])
    ):
        raise ValueError(
            "confidence with unit 'velocity' should be between 0 and 127. "
            + "Found values outside [0, 127]."
        )


def validate_voicing(voicing, voicing_unit):
    """Validate if voicing is well-formed.

    Args:
        voicing (np.ndarray): an array of voicing values
        voicing_unit (str): one of VOICING_UNITS

    Raises:
        ValueError: if voicing values are incompatible with the unit

    """
    validate_unit(voicing_unit, VOICING_UNITS)

    voicing_shape = np.shape(voicing)
    if len(voicing_shape) != 1:
        raise ValueError(f"voicings should be 1d, but array has shape {voicing_shape}")

    if voicing_unit == "continuous" and (
        any([c < 0 for c in voicing]) or any([c > 1 for c in voicing])
    ):
        raise ValueError(
            "voicing with unit 'continuous' should be between 0 and 1. "
            + "Found values outside [0, 1]."
        )

    if voicing_unit == "binary" and any([c not in [0, 1] for c in voicing]):
        raise ValueError(
            "voicing with unit 'binary' should only have values of 0 or 1. "
            + "Found non-binary values."
        )


def validate_pitches(pitches, pitch_unit):
    """Validate if pitches are well-formed.

    Args:
        pitches (np.ndarray): an array of pitch values
        pitch_unit (str): pitch unit, one of PITCH_UNITS

    Raises:
        ValueError: if pitches do not correspond to the unit

    """
    validate_unit(pitch_unit, PITCH_UNITS)
    if pitch_unit in ["hz", "midi"] and np.any(
        [np.any(np.array(p) < 0) for p in pitches]
    ):
        raise ValueError(
            "pitches should be positive numbers. "
            + "Unvoiced frames should be indicated using the confidence field, "
            + "rather than negative pitch values."
        )

    if pitch_unit == "midi" and np.any([np.any(np.array(p) > 127) for p in pitches]):
        raise ValueError("pitches in midi format cannot be larger than 127. ")

    if pitch_unit in ["pc", "note_name"]:
        try:
            librosa.note_to_midi(pitches)
        except:
            raise ValueError("invalid format for unit pc or note_name")


def validate_chord_labels(chords, chord_unit):
    """Validate that chord labels conform to chord_unit namespace

    Args:
        chords (list): list of chord labels as strings
        chord_unit (str): chord namespace, e.g. "harte"

    Raises:
        ValueError: If chords don't conform to namespace

    """
    validate_unit(chord_unit, CHORD_UNITS)

    if chord_unit in ["harte", "jams"]:
        if chord_unit == "harte":
            pattern = namespace("chord_harte")["properties"]["value"]["pattern"]
        elif chord_unit == "jams":
            pattern = namespace("chord")["properties"]["value"]["pattern"]

        matches = [re.match(pattern, c) for c in chords]
        if not all(matches):
            non_matches = [c for (c, m) in zip(chords, matches) if not m]
            raise ValueError(
                "chords {} don't conform to chord_unit {}".format(
                    non_matches, chord_unit
                )
            )


def validate_key_labels(keys, key_unit):
    """Validate that key labels conform to key_unit namespace

    Args:
        keys (list): list of key labels as strings
        key_unit (str): key namespace, e.g. "harte"

    Raises:
        ValueError: If keys don't conform to namespace

    """
    validate_unit(key_unit, KEY_UNITS)

    if key_unit == "key_mode":
        pattern = namespace("key_mode")["properties"]["value"]["pattern"]
        matches = [re.match(pattern, c) for c in keys]
        if not all(matches):
            non_matches = [k for (k, m) in zip(keys, matches) if not m]
            raise ValueError(
                "keys {} don't conform to key_unit key-mode".format(non_matches)
            )


def validate_times(times, time_unit):
    """Validate if times are well-formed.

    If times is None, validation passes automatically

    Args:
        times (np.ndarray): an array of time stamps
        time_unit (str): one of TIME_UNITS

    Raises:
        ValueError: if times have negative values or are non-increasing

    """
    if times is None:
        return

    validate_unit(time_unit, TIME_UNITS)

    time_shape = np.shape(times)
    if len(time_shape) != 1:
        raise ValueError(f"Times should be 1d, but array has shape {time_shape}")

    if (times < 0).any():
        raise ValueError("times should be positive numbers")

    if (times[1:] - times[:-1] <= 0).any():
        raise ValueError("times should be strictly increasing")


def validate_intervals(intervals, interval_unit):
    """Validate if intervals are well-formed.

    If intervals is None, validation passes automatically

    Args:
        intervals (np.ndarray): (n x 2) array
        interval_unit (str): interval unit, one of TIME_UNITS

    Raises:
        ValueError: if intervals have an invalid shape, have negative values
        or if end times are smaller than start times.

    """
    if intervals is None:
        return

    validate_unit(interval_unit, TIME_UNITS)

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


def validate_unit(unit, unit_values, allow_none=False):
    """Validate that the given unit is one of the allowed unit values.

    Args:
        unit (str): the unit name
        unit_values (dict): dictionary of possible unit values
        allow_none (bool): if true, allows unit=None to pass validation

    Raises:
        ValueError: If the given unit is not one of the allowed unit valuess
    """
    if allow_none and not unit:
        return

    if unit not in unit_values:
        raise ValueError("unit={} is not one of {}".format(unit, unit_values))
