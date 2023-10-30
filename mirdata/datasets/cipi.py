"""Can I play it? (CIPI) Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The "Can I Play It?" (CIPI) dataset is a specialized collection of 652 classical piano scores, provided in a
    machine-readable MusicXML format and accompanied by integer-based difficulty levels ranging from 1 to 9, as
    verified by expert pianists. Developed by the Music Technology Group in Barcelona, this dataset focuses
    exclusively on classical piano music, offering a rich resource for music researchers, educators, and students.

    The CIPI dataset facilitates various applications such as the study of musical complexity, the selection of
    appropriately leveled pieces for students, and general research in music education. The dataset, alongside
    embeddings of multiple dimensions of difficulty, has been made publicly available to encourage ongoing innovation
    and collaboration within the music education and research communities.
"""
import json
import logging
import os
import pickle
from typing import Optional, TextIO, List

from deprecated.sphinx import deprecated

from mirdata import core, io, jams_utils, download_utils

try:
    import music21
except ImportError:
    logging.error(
        "In order to use cipi you must have music21 installed. "
        "Please reinstall mirdata using `pip install 'mirdata[cipi]'"
    )
    raise ImportError


BIBTEX = """
@article{Ramoneda2024,
  author    = {Pedro Ramoneda and Dasaem Jeong and Vsevolod Eremenko and Nazif Can Tamer and Marius Miron and Xavier Serra},
  title     = {Combining Piano Performance Dimensions for Score Difficulty Classification},
  journal   = {Expert Systems with Applications},
  volume    = {238},
  pages     = {121776},
  year      = {2024},
  doi       = {10.1016/j.eswa.2023.121776},
  url       = {https://doi.org/10.1016/j.eswa.2023.121776}
}"""

INDEXES = {
    "default": "1.0",
    "test": "1.0",
    "1.0": core.Index(filename="cipi_1.0.json"),
}


LICENSE_INFO = (
    "Creative Commons Attribution Non Commercial Share Alike 4.0 International."
)


class Track(core.Track):
    """Can I play it? (CIPI) track class

    Args:
        track_id (str): track id of the track

    Attributes:
        title (str): title of the track
        book (str): book of the track
        URI (str): URI of the track
        composer (str): name of the author of the track
        track_id (str): track id
        musicxml_paths (list): path to musicxml score. If the music piece contains multiple movents the list will contain multiple paths.
        difficulty annotation (str): annotated difficulty

    Cached Properties:
        Fingering (tuple): fingering features from technique dimension computed with ArGNN fingering model. Return of two elements the embeddings of the right hand and the ones of the left hand.
        Expressiviness (list): expressiviness features from sound dimension computed with virtuosoNet model.
        Notes (list): note features from notation dimension.
        scores (list[music21.stream.Score]): music21 scores. If the work is splited in several movements the list will contain multiple scores.
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

    @property
    def title(self) -> str:
        return self._track_metadata.get("title")

    @property
    def book(self) -> str:
        return self._track_metadata.get("book")

    @property
    def URI(self) -> str:
        return self._track_metadata.get("URI")

    @property
    def composer(self) -> str:
        return self._track_metadata.get("composer")

    @property
    def track_id(self) -> str:
        return self._track_metadata.get("track_id")

    @property
    def musicxml_paths(self) -> List[str]:
        return list(self._track_metadata.get("musicxml_paths").values())

    @property
    def difficulty_annotation(self) -> str:
        return self._track_metadata.get("difficulty_annotation")

    @core.cached_property
    def scores(self) -> music21.stream.Score:
        return [load_score(path) for path in self.musicxml_paths]

    @core.cached_property
    def fingering(self) -> tuple:
        return _load_embedding(self.get_path("rh_fingering")), _load_embedding(self.get_path("lh_fingering"))

    @core.cached_property
    def expressiviness(self) -> list[list]:
        return _load_embedding(self.get_path("expressiviness"))

    @core.cached_property
    def notes(self) -> list[list]:
        return _load_embedding(self.get_path("notes"))

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            metadata={
                "title": self.title,
                "book": self.book,
                "URI": self.URI,
                "composer": self.composer,
                "track_id": self.track_id,
                "musicxml_paths": self.musicxml_paths,
                "difficulty_annotation": self.difficulty_annotation,
            }
        )




def _load_embedding(fpath):
    """Get the track's data in jams format

    Args:
        fpath (str or file-like): path to score file

    Returns:
        jams.JAMS: the track's data in jams format

    """
    with open(fpath, 'rb') as f:
        embedding = pickle.load(f)
    return embedding



def load_score(fhandle: TextIO):
    """Load cipi score in music21 stream

    Args:
        fhandle (str or file-like): path to MusicXML score

    Returns:
        music21.stream.Score: score in music21 format
    """
    score = music21.converter.parse(fhandle.name)
    return score





@deprecated(
    reason="convert_and_save_to_midi is deprecated and will be removed in a future version",
    version="0.3.4",
)
@io.coerce_to_string_io
def convert_and_save_to_midi(fpath: TextIO):
    """convert to midi file and return the midi path

    Args:
        fpath (str or file-like): path to score file

    Returns:
        str: midi file path

    """
    midi_path = os.path.splitext(fpath.name)[0] + ".midi"
    score, _ = music21.converter.parse(fpath)
    score.write("midi", fp=midi_path)
    return midi_path


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Can I play it? (CIPI) dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            name="cipi",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "index.json")
        try:
            with open(metadata_path, "r") as fhandle:
                return json.load(fhandle)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
