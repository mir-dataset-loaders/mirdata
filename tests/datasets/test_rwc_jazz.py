import os
import numpy as np
import pretty_midi

from mirdata.datasets import rwc_jazz
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "RWC_J004"
    data_home = os.path.normpath("tests/resources/mir_datasets/rwc_jazz")
    dataset = rwc_jazz.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "RWC_J004",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "RWC-J/RWC_J004.wav",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2/01_annotations_preprocessed/beats/RWC-J/RWC_J004.csv",
        ),
        "midi_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_jazz/"),
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2/01_annotations_preprocessed/MIDI_aligned/RWC-J/RWC_J004.mid",
        ),
        "piece_number": "4",
        "cd_number": "1",
        "track_number": "4",
        "title": "Crescent Serenade (Piano Solo)",
        "artist": "Makoto Nakamura",
        "singer_information": "",
        "singing_language": "",
        "tempo": "",
        "variation": "Instrumentation 1",
        "live_instrument": "Pf",
        "drum_information": "",
        "composer": "",
        "composition_type": "",
        "main_genre": "Jazz",
        "sub_genre": "",
        "audio_start": "0.0241",
        "audio_end": "162.2121",
        "duration": "165.38698412698412",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "midi": pretty_midi.PrettyMIDI,
        "notes": annotations.NoteData,
        "audio": tuple
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 1,)
    # test NoteData
    assert isinstance(track.notes, annotations.NoteData)
    assert len(track.notes.intervals) > 0




def test_load_beats():
    beats_path = (
        "tests/resources/mir_datasets/rwc_jazz/"
        + "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2/01_annotations_preprocessed/beats/RWC-J/RWC_J004.csv"
    )
    beat_data = rwc_jazz.load_beats(beats_path)

    # check types
    assert type(beat_data) is annotations.BeatData
    assert type(beat_data.times) is np.ndarray
    assert type(beat_data.positions) is np.ndarray

    # check values
    assert np.array_equal(
        beat_data.times[:4], np.array([0.050, 0.455, 0.860, 1.265])
    )
    assert np.array_equal(beat_data.positions[:4], np.array([1, 2, 3, 4]))


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/rwc_jazz"
    dataset = rwc_jazz.Dataset(data_home, version="test")
    metadata = dataset._metadata

    assert "RWC_J004" in metadata
    assert metadata["RWC_J004"]["piece_number"] == "4"
    assert metadata["RWC_J004"]["cd_number"] == "1"
    assert metadata["RWC_J004"]["track_number"] == "4"
    assert metadata["RWC_J004"]["title"] == "Crescent Serenade (Piano Solo)"
    assert metadata["RWC_J004"]["artist"] == "Makoto Nakamura"
    assert metadata["RWC_J004"]["variation"] == "Instrumentation 1"
    assert metadata["RWC_J004"]["live_instrument"] == "Pf"
    assert metadata["RWC_J004"]["main_genre"] == "Jazz"
    assert metadata["RWC_J004"]["audio_start"] == "0.0241"
    assert metadata["RWC_J004"]["audio_end"] == "162.2121"
    assert metadata["RWC_J004"]["duration"] == "165.38698412698412"


def test_load_metadata_else(tmp_path):
    data_home = tmp_path / "rwc_jazz"
    metadata_dir = (
        data_home / "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2"
    )
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.csv"

    metadata_file.write_text(
        "RWCID;CollID;PieceNo;CDNo;TrackNo;Title;Artist;SingerInformation;"
        "SingingLanguage;Tempo;Variation;LiveInstruments;DrumInformation;"
        "Composer;CompositionType;GenreMain;GenreSub;audio_start;audio_end;duration\n"
        "RWC_C001;C;1;1;1;Test;Test Artist;;;;;;;;;Classical;;0.1;10.0;10.0\n",
        encoding="utf-8",
    )

    dataset = rwc_jazz.Dataset(str(data_home), version="test")
    metadata = dataset._metadata

    assert metadata == {}
