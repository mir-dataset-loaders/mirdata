"""

"""

import csv
import logging
import os
import shutil
import re
from typing import BinaryIO, TextIO, Optional, Tuple

import librosa
import numpy as np

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import core
from mirdata import annotations
from mirdata import io

BIBTEX = """@inproceedings{burgoyne_billboard,
author = {Burgoyne, John Ashley and Wild, Jonathan and Fujinaga, Ichiro},
year = {2011},
title = {An {Expert} {Ground} {Truth} {Set} for {Audio} {Chord} {Recognition} and {Music} {Analysis}},
booktitle={Proceedings of the 12th International Society for Music Information Retrieval Conference, ISMIR}
}"""
REMOTES = {
    "index": download_utils.RemoteFileMetadata(
        filename="billboard-2.0-index.csv",
        url="https://www.dropbox.com/s/o0olz0uwl9z9stb/billboard-2.0-index.csv?dl=1",
        checksum="c47d304c212725998839cf9bb1a417aa",
        destination_dir="annotation",
    ),
    "annotation_salami": download_utils.RemoteFileMetadata(
        filename="billboard-2.0-salami_chords.tar.gz",
        url="https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1",
        checksum="6954a6fad962a111e69c9c80cb87d3a5",
        destination_dir="annotation",
    ),
    "annotation_lab": download_utils.RemoteFileMetadata(
        filename="billboard-2.0.1-lab.tar.gz",
        url="https://www.dropbox.com/s/t390alzrkx0c9yt/billboard-2.0.1-lab.tar.gz?dl=1",
        checksum="a7b1fa6a7e454bf73ced7c29207aa597",
        destination_dir="annotation",
    ),
    "annotation_mirex13": download_utils.RemoteFileMetadata(
        filename="billboard-2.0.1-mirex.tar.gz",
        url="https://www.dropbox.com/s/fg8lvy79o7etiyc/billboard-2.0.1-mirex.tar.gz?dl=1",
        checksum="97e5754699f3b45aa5cc70d8a7611c54",
        destination_dir="annotation",
    ),
    "annotation_chordino": download_utils.RemoteFileMetadata(
        filename="billboard-2.0-chordino.tar.gz",
        url="https://www.dropbox.com/s/e9dm23vbawg9dsw/billboard-2.0-chordino.tar.gz?dl=1",
        checksum="530218e8d7077bbd4b08b45f447f5e8f",
        destination_dir="annotation",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International Public License."


class Track(core.Track):
    """Billboard Dataset Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path of the audio file
        dynamics (str): dynamics abbreviation. Ex: pp, mf, ff, etc.
        dynamics_id (int): pp=0, p=1, mf=2, f=3, ff=4
        family (str): instrument family encoded by its English name
        instance_id (int): instance ID. Either equal to 0, 1, 2, or 3.
        instrument_abbr (str): instrument abbreviation
        instrument_full (str): instrument encoded by its English name
        is_resampled (bool): True if this sample was pitch-shifted from a neighbor; False if it was genuinely recorded.
        pitch (str): string containing English pitch class and octave number
        pitch_id (int): MIDI note index, where middle C ("C4") corresponds to 60
        string_id (NoneType): string ID. By musical convention, the first
            string is the highest. On wind instruments, this is replaced by `None`.
        technique_abbr (str): playing technique abbreviation
        technique_full (str): playing technique encoded by its English name
        track_id (str): track id

    """

    def __init__(
        self,
        track_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.audio_path = os.path.join(self._data_home, self._track_paths["audio"][0])

        self.salami_path = os.path.join(self._data_home, self._track_paths["salami"][0])
        self.lab_full_path = os.path.join(
            self._data_home, self._track_paths["lab_full"][0]
        )
        self.lab_majmin7_path = os.path.join(
            self._data_home, self._track_paths["lab_majmin7"][0]
        )
        self.lab_majmin7inv_path = os.path.join(
            self._data_home, self._track_paths["lab_majmin7inv"][0]
        )
        self.lab_majmin_path = os.path.join(
            self._data_home, self._track_paths["lab_majmin"][0]
        )
        self.lab_majmininv_path = os.path.join(
            self._data_home, self._track_paths["lab_majmininv"][0]
        )
        self.bothchroma_path = os.path.join(
            self._data_home, self._track_paths["bothchroma"][0]
        )
        self.tuning_path = os.path.join(self._data_home, self._track_paths["tuning"][0])

    @property
    def chart_date(self):
        return self._track_metadata.get("chart_date")

    @property
    def target_rank(self):
        return self._track_metadata.get("target_rank")

    @property
    def actual_rank(self):
        return self._track_metadata.get("actual_rank")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def peak_rank(self):
        return self._track_metadata.get("peak_rank")

    @property
    def weeks_on_chart(self):
        return self._track_metadata.get("weeks_on_chart")

    @core.cached_property
    def chords(self):
        return {
            "full": load_chords(self.lab_full_path),
            "majmin7": load_chords(self.lab_majmin7_path),
            "majmin7inv": load_chords(self.lab_majmin7inv_path),
            "majmin": load_chords(self.lab_majmin_path),
            "majmininv": load_chords(self.lab_majmininv_path),
        }

    @core.cached_property
    def salami_metadata(self):
        return _parse_salami_metadata(
            os.path.join(self._data_home, self._track_paths["salami"][0])
        )

    @core.cached_property
    def chroma(self):
        with open(self.bothchroma_path, "r") as f:
            return np.array([l for l in csv.reader(f)])

    @core.cached_property
    def tuning(self):
        with open(self.tuning_path, "r") as f:
            return np.array([l for l in csv.reader(f)])

    @core.cached_property
    def sections(self):
        return load_sections(
            os.path.join(self._data_home, self._track_paths["salami"][0]),
            section_type="letter",
        )

    @core.cached_property
    def named_sections(self):
        return load_sections(
            os.path.join(self._data_home, self._track_paths["salami"][0]),
            section_type="name",
        )

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._track_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a TinySOL audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_chords(fhandle: TextIO):
    """Private function to load LAB format chord data from a file
    Args:
        chords_path (str):
    """
    start_times = []
    end_times = []
    chords = []
    for l in fhandle:
        l = l.rstrip()
        if l:
            start, end, label = l.split("\t")
            start_times.append(float(start))
            end_times.append(float(end))
            chords.append(label)

    chord_data = annotations.ChordData(np.array([start_times, end_times]).T, chords)
    return chord_data


@io.coerce_to_string_io
def _parse_timed_sections(fhandle: TextIO):
    salami = _parse_salami(fhandle)
    timed_sections = _timed_sections(salami)
    return timed_sections


def load_sections(fpath, section_type: str) -> annotations.SectionData:
    """Private function to load SALAMI format sections data from a file
    Args:
        sections_path (str):
    """

    timed_sections = _parse_timed_sections(fpath)
    # Clean sections
    timed_sections_clean = [ts for ts in timed_sections if ts["section"] is not None]

    start_times = []
    end_times = []
    sections = []

    if section_type == "letter":
        section_label_idx = 0
    elif section_type == "name":
        section_label_idx = 1
    else:
        raise ValueError("This section type is not available.")

    for idx, ts in enumerate(timed_sections_clean):
        if idx < len(timed_sections_clean) - 1:
            start_times.append(timed_sections_clean[idx]["time"])
            end_times.append(timed_sections_clean[idx + 1]["time"])
            sections.append(timed_sections_clean[idx]["section"][section_label_idx])
        else:
            start_times.append(timed_sections_clean[idx]["time"])
            end_times.append(timed_sections[-1]["time"])  # end of song
            sections.append(timed_sections_clean[idx]["section"][section_label_idx])

    section_data = annotations.SectionData(
        np.array([start_times, end_times]).T, sections
    )
    return section_data


def _parse_salami(fhandle):
    """
    Author:
        Brian Whitman
        brian@echonest.com
        https://gist.github.com/bwhitman/11453443
    Parse a salami_chords.txt file and return a dict with all the stuff innit
    """
    s = fhandle.read().split("\n")
    o = {}
    for x in s:
        if x.startswith("#"):
            if x[2:].startswith("title:"):
                o["title"] = x[9:]
            if x[2:].startswith("artist:"):
                o["artist"] = x[10:]
            if x[2:].startswith("metre:"):
                o["meter"] = o.get("meter", []) + [x[9:]]
            if x[2:].startswith("tonic:"):
                o["tonic"] = o.get("tonic", []) + [x[9:]]
        elif len(x) > 1:
            spot = x.find("\t")
            if spot > 0:
                time = float(x[0:spot])
                event = {}
                event["time"] = time
                rest = x[spot + 1 :]
                items = rest.split(", ")
                for i in items:
                    chords = re.findall(r"(?=\| (.*?) \|)", i)
                    section = i.split("|")
                    if len(section) == 1 and not ("(" in section or ")" in section):
                        event["section"] = section[0]
                    if len(chords):
                        event["chords"] = chords
                    else:
                        event["notes"] = event.get("notes", []) + [i]
                o["events"] = o.get("events", []) + [event]
    return o


def _timed_sections(parsed):
    """
    Author:
        Brian Whitman
        brian@echonest.com
        https://gist.github.com/bwhitman/11453443
    Given a salami parse return a list of parsed chords with timestamps & deltas
    """
    timed_sections = []
    tic = 0
    for i, e in enumerate(parsed["events"]):
        sections = []
        try:
            dt = parsed["events"][i + 1]["time"] - e["time"]
        except IndexError:
            dt = 0

        section = None
        if e.get("notes"):
            if len(e.get("notes")) > 1:
                section = (e.get("notes")[0], e.get("notes")[1])
            sections.append(section)

        tic = e["time"]
        if len(sections):
            seconds_per_chord = dt / float(len(sections))
            for c in sections:
                timed_sections.append(
                    {"time": tic, "section": c, "length": seconds_per_chord}
                )
                tic = tic + seconds_per_chord
    return timed_sections


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The McGill Billboard dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="billboard",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(
            self.data_home, "annotation", "billboard-2.0-index.csv"
        )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            next(reader, None)
            raw_data = []
            for line in reader:
                if line != []:
                    raw_data.append(line)

        metadata_index = {}
        for line in raw_data:
            track_id = line[0]
            metadata_index[track_id] = {
                "chart_date": line[1],
                "target_rank": int(line[2]) if line[2] else None,
                "actual_rank": int(line[3]) if line[3] else None,
                "title": line[4],
                "artist": line[5],
                "peak_rank": int(line[6]) if line[6] else None,
                "weeks_on_chart": int(line[7]) if line[7] else None,
            }
        return metadata_index

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    
    def download(self, force_overwrite=False, cleanup=False):
        """Download the dataset

        Args:
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        """
        info_message = """
            Unfortunately the audio files of the McGill-Billboard dataset are not available
            for download. If you have the McGill-Billboard dataset, place the contents into a
            folder called McGill-Billboard with the following structure:
                > billboard/
                    > annotation/
                    > audio/
            and copy the billboard folder to {}
        """.format(
            self.data_home
        )

        annotations_dir = os.path.join(self.data_home, "annotation")
        sub_dir = os.path.join(annotations_dir, "McGill-Billboard")

        # download_utils.downloader(
        #     self.data_home,
        #     remotes=REMOTES,
        #     info_message=info_message,
        #     force_overwrite=force_overwrite,
        #     cleanup=cleanup,
        # )

        if os.path.exists(sub_dir):
            for f in os.listdir(sub_dir):
                    shutil.move(os.path.join(sub_dir, f), annotations_dir)
            shutil.rmtree(sub_dir)