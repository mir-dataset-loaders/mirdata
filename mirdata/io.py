import functools
import io
from typing import BinaryIO, Callable, Optional, TextIO, Tuple, TypeVar, Union

import librosa
import numpy as np
import pretty_midi

from mirdata import annotations

T = TypeVar("T")  # Can be anything


def coerce_to_string_io(
    func: Callable[[TextIO], T]
) -> Callable[[Optional[Union[str, TextIO]]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(file_path_or_obj: Optional[Union[str, TextIO]]) -> Optional[T]:
        if not file_path_or_obj:
            return None
        if isinstance(file_path_or_obj, str):
            with open(file_path_or_obj) as f:
                return func(f)
        elif isinstance(file_path_or_obj, io.StringIO):
            return func(file_path_or_obj)
        else:
            raise ValueError(
                "Invalid argument passed to {}, argument has the type {}",
                func.__name__,
                type(file_path_or_obj),
            )

    return wrapper


def coerce_to_bytes_io(
    func: Callable[[BinaryIO], T]
) -> Callable[[Optional[Union[str, BinaryIO]]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(file_path_or_obj: Optional[Union[str, BinaryIO]]) -> Optional[T]:
        if not file_path_or_obj:
            return None
        if isinstance(file_path_or_obj, str):
            with open(file_path_or_obj, "rb") as f:
                return func(f)
        elif isinstance(file_path_or_obj, io.BytesIO):
            return func(file_path_or_obj)
        else:
            raise ValueError(
                "Invalid argument passed to {}, argument has the type {}",
                func.__name__,
                type(file_path_or_obj),
            )

    return wrapper


# @coerce_to_bytes_io
# def load_audio(
#     fhandle: BinaryIO, samplerate: Optional[float], mono: bool
# ) -> Tuple[np.ndarray, float]:
#     """Load an audio file.

#     Args:
#         fhandle (str or file-like): File-like object or path to audio file
#         samplerate (float or None): Sample rate at which to load file, or None
#             which loads the native sample rate of the audio file
#         mono (bool): if True, loads audio as mono, averaging multiple channels
#             if present. if False, loads audio with the native number of channels.

#     Returns:
#         * np.ndarray - the mono audio signal
#         * float - The sample rate of the audio file

#     """
#     return librosa.load(fhandle, sr=samplerate, mono=mono)


@coerce_to_bytes_io
def load_midi(fhandle: BinaryIO) -> pretty_midi.PrettyMIDI:
    """Load a midi file.

    Args:
        fhandle (str or file-like): File-like object or path to midi file

    Returns:
        pretty_midi.PrettyMIDI: pretty_midi object

    """
    return pretty_midi.PrettyMIDI(fhandle)


def load_notes_from_midi(
    midi_path: Optional[Union[str, BinaryIO]] = None,
    midi: Optional[pretty_midi.PrettyMIDI] = None,
) -> annotations.NoteData:
    """Load note data from a midi file or

    Args:
        midi_path (str or None): path to midi file or None
        midi (pretty_midi.PrettyMIDI or None): pre-loaded midi object or None
            if None, the midi object is loaded using midi_path

    Returns:
        NoteData: note annotations

    """
    if not midi and not midi_path:
        raise ValueError("At least one of midi_path or midi must be provided")
    elif not midi:
        midi = load_midi(midi_path)

    intervals = []
    pitches = []
    confidence = []
    for instrument in midi.instruments:  # type: ignore
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            pitches.append(librosa.midi_to_hz(note.pitch))
            confidence.append(note.velocity / 127.0)
    return annotations.NoteData(
        np.array(intervals), np.array(pitches), np.array(confidence)
    )
