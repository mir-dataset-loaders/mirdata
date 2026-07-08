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

    Every performance is provided as (a) a MIDI file captured by the Bösendorfer
    reproducing piano and (b) a note-wise score-to-performance alignment in
    ``.match`` format (v1.0.0). MusicXML symbolic scores of the four excerpts are
    also provided, one per piece.

    This loader wraps the version curated by the Institute of Computational Perception
    (CPJKU, JKU Linz) at https://github.com/CPJKU/vienna4x22. The data is released under
    the Creative Commons Attribution 4.0 International License (CC BY 4.0). Score, MIDI
    performance, and match files are parsed with `partitura <https://github.com/CPJKU/partitura>`_.

    References:

    * W. Goebl. Numerisch-klassifikatorische Interpretationsanalyse mit dem
      "Bösendorfer Computerflügel". PhD thesis, University of Vienna, 1999.
    * F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, M. Sakai.
      "The match file format: Encoding Alignments between Scores and Performances", 2022.

"""

import logging
from typing import BinaryIO, TextIO

from mirdata import core, download_utils, io

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
        url="https://zenodo.org/records/21160492/files/vienna4x22_index_1.0.json?download=1",
        checksum="989a919bc6102020bbf2dca44bea3d23",
    ),
    "sample": core.Index(filename="vienna4x22_index_1.0_sample.json"),
}

# The CPJKU/vienna4x22 GitHub repo has no tagged release; pin to a specific
# commit tarball so the checksum is stable.
# ponytail: pinned to master @ 1033ade; bump when upstream cuts a release.
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="vienna4x22-1033ade.tar.gz",
        url=(
            "https://codeload.github.com/CPJKU/vienna4x22/tar.gz/"
            "1033ade0899bfd03a89f370c9ad5d8443ddccd3e"
        ),
        checksum="a441555d302d57b1ab0922b342769184",
        unpack_directories=["vienna4x22-1033ade0899bfd03a89f370c9ad5d8443ddccd3e"],
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
            hand-corrected. Datasets with automatic alignment (e.g. parangonar)
            will use ``"automatic"``.
        score_path (str): path to the piece's MusicXML score.
        performance_path (str): path to the performance MIDI file.
        match_path (str): path to the score/performance ``.match`` alignment file.

    Cached Properties:
        score (partitura.score.Score): score parsed with partitura. Access the full
            partitura API for part structure, key/time signatures, tempo markings,
            etc. (e.g. ``track.score[0].key_sigs``).
        performance (partitura.performance.Performance): performance parsed with partitura.
        match (tuple): ``(performance, alignment, score)`` as returned by
            :func:`partitura.load_match` with ``create_score=True``. Each entry in
            the alignment list carries a ``"label"`` key: ``"match"``, ``"deletion"``
            (score note not played), ``"insertion"`` (extra performance note), or
            ``"ornament"``.
        note_array (numpy.ndarray): score note array with default fields. For custom
            fields (pitch spelling, metrical position, grace notes, etc.) call
            ``track.score.note_array(**kwargs)`` directly.
        performance_note_array (numpy.ndarray): performance note array with default
            fields. For custom fields call ``track.performance.note_array(**kwargs)``.

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)
        self.score_path = self.get_path("score")
        self.performance_path = self.get_path("performance")
        self.match_path = self.get_path("match")
        piece, pianist_id = track_id.rsplit("_p", 1)
        self.piece = piece
        self.pianist_id = pianist_id
        self.alignment_quality = "manual"

    @core.cached_property
    def score(self):
        return load_score(self.score_path)

    @core.cached_property
    def performance(self):
        return load_performance(self.performance_path)

    @core.cached_property
    def match(self):
        return load_match(self.match_path)

    @core.cached_property
    def note_array(self):
        return self.score.note_array()

    @core.cached_property
    def performance_note_array(self):
        return self.performance.note_array()


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
