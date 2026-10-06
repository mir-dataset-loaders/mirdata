"""Vienna 4x22 Piano Corpus Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Vienna 4x22 Piano Corpus is a symbolic dataset of piano performances originally
    compiled by Werner Goebl (1999). Twenty-two professional pianists each performed the
    same four excerpts from the Classic-Romantic repertoire on a Bösendorfer SE290 computer-
    controlled Imperial grand piano, yielding 88 performances in total.

    The four excerpts are:

    * ``Chopin_op10_no3`` — 21 bars of Chopin Op. 10 No. 3
    * ``Chopin_op38``    — 45 bars of Chopin Op. 38 (Ballade No. 2)
    * ``Mozart_K331_1st-mov`` — 36 bars (exposition) of Mozart K. 331, 1st mov.
    * ``Schubert_D783_no15``  — full 32 bars of Schubert D. 783 No. 15

    Every performance is provided as (a) a stereo WAV audio recording, (b) a MIDI file
    captured by the Bösendorfer reproducing piano, and (c) a note-wise score-to-performance
    alignment in ``.match`` format (v1.0.0). MusicXML symbolic scores of the four excerpts
    are also provided, one per piece.

    Symbolic data (MIDI, MusicXML, match files) are curated by the Institute of
    Computational Perception (CPJKU, JKU Linz) at https://github.com/CPJKU/vienna4x22.
    Audio recordings are hosted separately by the University of Music and Performing Arts
    Vienna (mdw) at https://datasets.mdw.ac.at (DOI: 10.21939/4X22).
    All data is released under the Creative Commons Attribution 4.0 International License
    (CC BY 4.0). Score, MIDI performance, and match files are parsed with
    `partitura <https://github.com/CPJKU/partitura>`_.

    Note IDs: score note IDs come from the MusicXML file. MIDI files carry no note
    IDs, so every performance note is given the ID that the ``.match`` file assigns
    to it. Within a track, a note has the same ID in ``score``, ``performance``,
    ``match`` and every derived note array, so the alignment can be joined directly.

    References:

    * W. Goebl. Numerisch-klassifikatorische Interpretationsanalyse mit dem
      "Bösendorfer Computerflügel". PhD thesis, University of Vienna, 1999.
    * F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, M. Sakai.
      "The match file format: Encoding Alignments between Scores and Performances", 2022.

"""

import logging
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np

from mirdata import annotations, core, download_utils, io

try:
    import partitura
except ImportError:
    logging.error(
        "In order to use vienna4x22 you must have partitura installed. "
        "Please reinstall mirdata using `pip install 'mirdata[vienna4x22]'`."
    )
    raise ImportError


BIBTEX = """
@phdthesis{goebl1999vienna4x22,
  author = {Goebl, Werner},
  title  = {Numerisch-klassifikatorische {I}nterpretationsanalyse mit dem
            "{B}\\"{o}sendorfer {C}omputerfl\\"{u}gel"},
  school = {University of Vienna},
  year   = {1999}
}
@inproceedings{foscarin2022match,
  author = {Foscarin, Francesco and McLeod, Andrew and Rigaux, Philippe
            and Jacquemard, Florent and Sakai, Masahiko},
  title  = {The match file format: Encoding Alignments between Scores and Performances},
  booktitle = {Proc. of the Music Encoding Conference (MEC)},
  year   = {2022}
}"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="vienna4x22_index_1.0.json",
        url="https://zenodo.org/records/21443270/files/vienna4x22_index_1.0.json?download=1",
        checksum="5e127f886fc5b83d7c83623152dcc160",
    ),
    "sample": core.Index(filename="vienna4x22_index_1.0_sample.json"),
}

# The CPJKU/vienna4x22 GitHub repo has no tagged release; pin to a specific
# commit tarball so the checksum is stable.
# ponytail: pinned to master @ 1033ade; bump when upstream cuts a release.
REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="vienna4x22-1033ade.tar.gz",
        url=(
            "https://codeload.github.com/CPJKU/vienna4x22/tar.gz/"
            "1033ade0899bfd03a89f370c9ad5d8443ddccd3e"
        ),
        checksum="a441555d302d57b1ab0922b342769184",
        unpack_directories=["vienna4x22-1033ade0899bfd03a89f370c9ad5d8443ddccd3e"],
    ),
    "audio": download_utils.RemoteFileMetadata(
        filename="vienna4x22-audio.zip",
        url=(
            "https://repo.mdw.ac.at/projects/IWK/"
            "the_vienna_4x22_piano_corpus/data/audio.zip"
        ),
        checksum="4fa07a425cd65e1f752ebc288767fff9",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International (CC BY 4.0)."

PIECES = (
    "Chopin_op10_no3",
    "Chopin_op38",
    "Mozart_K331_1st-mov",
    "Schubert_D783_no15",
)


class Track(core.Track):
    """Vienna 4x22 track class.

    A track represents one performance of one piece by one of the 22 pianists.
    The symbolic score (MusicXML) is shared across all 22 tracks of a piece.

    Args:
        track_id (str): track id, e.g. ``"Chopin_op10_no3_p01"``.

    Attributes:
        track_id (str): track id.
        piece (str): piece identifier, one of :data:`PIECES`.
        pianist_id (str): two-digit pianist id (``"01"`` .. ``"22"``).
        alignment_quality (str): ``"manual"`` — all Vienna 4x22 alignments were
            hand-corrected.
        audio_path (str or None): path to the stereo WAV recording (``None`` if
            the audio remote has not been downloaded).
        score_path (str): path to the piece's MusicXML score.
        performance_path (str): path to the performance MIDI file.
        match_path (str): path to the score/performance ``.match`` alignment file.

    Cached Properties:
        audio (tuple): ``(np.ndarray, float)`` stereo audio signal and sample rate.
        score (partitura.score.Score): score parsed with partitura. Access the full
            partitura API for part structure, key/time signatures, tempo markings,
            etc. (e.g. ``track.score[0].key_sigs``).
        performance (partitura.performance.Performance): the full MIDI performance
            (all notes and all pedal events), with note IDs taken from the ``.match``
            file. See :func:`align_performance_ids`.
        match (tuple): ``(performance, alignment, score)`` as returned by
            :func:`partitura.load_match` with ``create_score=True``. Each entry in
            the alignment list carries a ``"label"`` key: ``"match"``, ``"deletion"``
            (score note not played), ``"insertion"`` (extra performance note), or
            ``"ornament"``. The performance stored in the ``.match`` file has fewer
            pedal events than the MIDI file; use ``track.performance`` for pedals.
        score_note_array (numpy.ndarray): score note array with default fields. For
            custom fields (pitch spelling, metrical position, grace notes, etc.) call
            ``track.score.note_array(**kwargs)`` directly.
        performance_note_array (numpy.ndarray): performance note array with default
            fields, covering every performed note, sorted by onset, offset, then
            pitch. For custom fields call ``track.performance.note_array(**kwargs)``.
        matched_note_array (numpy.ndarray): one row per aligned score/performance
            note pair (alignment label ``"match"``), sorted by performance onset. See
            :func:`get_matched_note_array` for the fields. Unplayed score notes
            (deletions), performed notes absent from the score (insertions) and pedal
            events are not included; find them in ``track.match`` and
            ``track.performance``.
        performance_notes (annotations.NoteData): performed notes as intervals in
            seconds (offsets extended by the sustain pedal, as in
            ``performance_note_array["duration_sec"]``), MIDI pitches and MIDI
            velocities, e.g. for transcription against
            ``track.audio``. ``NoteData`` holds no note IDs, but its rows are in the
            same order as ``performance_note_array``: the ID of note ``i`` is
            ``track.performance_note_array["id"][i]``.

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)
        self.audio_path = self.get_path("audio")
        self.score_path = self.get_path("score")
        self.performance_path = self.get_path("performance")
        self.match_path = self.get_path("match")
        piece, pianist_id = track_id.rsplit("_p", 1)
        self.piece = piece
        self.pianist_id = pianist_id
        self.alignment_quality = "manual"

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        return load_audio(self.audio_path)

    @core.cached_property
    def score(self):
        return load_score(self.score_path)

    @core.cached_property
    def performance(self):
        performance = load_performance(self.performance_path)
        match_performance, _, _ = self.match
        return align_performance_ids(performance, match_performance)

    @core.cached_property
    def match(self):
        return load_match(self.match_path)

    @core.cached_property
    def score_note_array(self):
        return self.score.note_array()

    @core.cached_property
    def performance_note_array(self):
        # Same order as NoteData stores notes, so row i here is row i of
        # performance_notes.
        na = self.performance.note_array()
        offsets = na["onset_sec"] + na["duration_sec"]
        return na[np.lexsort((na["pitch"], offsets, na["onset_sec"]))]

    @core.cached_property
    def matched_note_array(self):
        _, alignment, _ = self.match
        return get_matched_note_array(
            self.score_note_array, self.performance_note_array, alignment
        )

    @core.cached_property
    def performance_notes(self):
        na = self.performance_note_array
        notes = annotations.NoteData(
            intervals=np.stack(
                [na["onset_sec"], na["onset_sec"] + na["duration_sec"]], axis=1
            ).astype(float),
            interval_unit="s",
            pitches=na["pitch"].astype(float),
            pitch_unit="midi",
            confidence=na["velocity"].astype(float),
            confidence_unit="velocity",
        )
        # NoteData sorts and deduplicates its notes; make sure that kept the rows
        # of performance_note_array, or row i would no longer have ID na["id"][i].
        if not np.array_equal(notes.pitches, na["pitch"]):
            raise ValueError(
                "performance_notes rows no longer line up with performance_note_array."
            )
        return notes


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a Vienna 4x22 WAV recording.

    Args:
        fhandle (str or file-like): path to a ``.wav`` file.

    Returns:
        * np.ndarray - stereo audio signal
        * float - sample rate

    """
    return librosa.load(fhandle, sr=None, mono=False)


@io.coerce_to_string_io
def load_score(fhandle: TextIO):
    """Load a Vienna 4x22 MusicXML score with partitura.

    Args:
        fhandle (str or file-like): path to a ``.musicxml`` file.

    Returns:
        partitura.score.Score: parsed score.

    """
    return partitura.load_score(fhandle.name)


@io.coerce_to_bytes_io
def load_performance(fhandle: BinaryIO):
    """Load a Vienna 4x22 performance MIDI with partitura.

    Args:
        fhandle (str or file-like): path to a ``.mid`` file.

    Returns:
        partitura.performance.Performance: parsed performance.

    """
    return partitura.load_performance_midi(fhandle.name)


@io.coerce_to_string_io
def load_match(fhandle: TextIO):
    """Load a Vienna 4x22 match file with partitura.

    Args:
        fhandle (str or file-like): path to a ``.match`` file.

    Returns:
        tuple: ``(performance, alignment, score)`` as returned by
        :func:`partitura.load_match` with ``create_score=True``.

    """
    return partitura.load_match(fhandle.name, create_score=True)


def align_performance_ids(performance, match_performance):
    """Give every note of a MIDI performance the ID its ``.match`` file assigns to it.

    MIDI files carry no note IDs, so partitura numbers MIDI notes itself, and that
    numbering does not always agree with the IDs the ``.match`` alignment refers to.
    Each MIDI note is looked up in the ``.match`` performance by
    ``(note_on_tick, midi_pitch)`` and takes over its ID; no partitura-generated ID
    survives. Pedal events are left untouched.

    Args:
        performance (partitura.performance.Performance): performance loaded from MIDI.
        match_performance (partitura.performance.Performance): performance loaded from
            the matching ``.match`` file.

    Returns:
        partitura.performance.Performance: ``performance``, relabelled in place.

    Raises:
        ValueError: if the two performances do not contain exactly the same notes,
            or a ``(note_on_tick, midi_pitch)`` key is not unique.

    """
    if performance is None or match_performance is None:
        return performance

    def key(note):
        return (note["note_on_tick"], note["midi_pitch"])

    match_notes = match_performance.performedparts[0].notes
    midi_notes = performance.performedparts[0].notes
    match_ids = {key(note): note["id"] for note in match_notes}
    midi_keys = {key(note) for note in midi_notes}
    if not (
        len(match_ids) == len(match_notes) == len(midi_notes) == len(midi_keys)
        and midi_keys == match_ids.keys()
    ):
        raise ValueError(
            "MIDI and .match performances do not contain the same notes with unique "
            "(onset tick, pitch) keys; cannot assign .match note IDs to the MIDI "
            "performance."
        )
    for note in midi_notes:
        note["id"] = match_ids[key(note)]
    return performance


def get_matched_note_array(score_note_array, performance_note_array, alignment):
    """Join score and performance note arrays along the ``"match"`` alignment pairs.

    Args:
        score_note_array (numpy.ndarray): score note array (needs ``id``, ``pitch``,
            ``onset_beat``, ``duration_beat``).
        performance_note_array (numpy.ndarray): performance note array whose IDs
            follow the ``.match`` file (needs ``id``, ``onset_sec``,
            ``duration_sec``, ``velocity``).
        alignment (list): alignment as returned by :func:`partitura.load_match`.

    Returns:
        numpy.ndarray: structured array with fields ``score_id``, ``performance_id``,
        ``score_pitch``, ``performance_pitch``, ``onset_beat``, ``duration_beat``, ``onset_sec``,
        ``duration_sec`` and ``velocity``, one row per matched pair, sorted by
        ``onset_sec`` then ``performance_pitch``. The two pitches can differ: a few
        ``.match`` files pair a score note with a note played in another octave
        (e.g. ``Chopin_op38_p05``, score ``n731``). ``duration_sec`` follows partitura and
        includes the sustain pedal.

    """
    score_idx = {note_id: i for i, note_id in enumerate(score_note_array["id"])}
    perf_idx = {note_id: i for i, note_id in enumerate(performance_note_array["id"])}
    pairs = [
        (score_idx[a["score_id"]], perf_idx[a["performance_id"]])
        for a in alignment
        if a["label"] == "match"
    ]
    s = score_note_array[[i for i, _ in pairs]]
    p = performance_note_array[[j for _, j in pairs]]

    matched = np.empty(
        len(pairs),
        dtype=[
            ("score_id", score_note_array.dtype["id"]),
            ("performance_id", performance_note_array.dtype["id"]),
            ("score_pitch", "i4"),
            ("performance_pitch", "i4"),
            ("onset_beat", "f4"),
            ("duration_beat", "f4"),
            ("onset_sec", "f4"),
            ("duration_sec", "f4"),
            ("velocity", "i4"),
        ],
    )
    matched["score_id"] = s["id"]
    matched["performance_id"] = p["id"]
    matched["score_pitch"] = s["pitch"]
    matched["performance_pitch"] = p["pitch"]
    matched["onset_beat"] = s["onset_beat"]
    matched["duration_beat"] = s["duration_beat"]
    matched["onset_sec"] = p["onset_sec"]
    matched["duration_sec"] = p["duration_sec"]
    matched["velocity"] = p["velocity"]
    return np.sort(matched, order=["onset_sec", "performance_pitch"], kind="stable")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The Vienna 4x22 Piano Corpus dataset."""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version=version,
            name="vienna4x22",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )
