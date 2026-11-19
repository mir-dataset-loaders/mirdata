import os
import numpy as np
import pretty_midi

from mirdata.datasets import rwc_classical
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "RWC_C003"
    data_home = os.path.normpath("tests/resources/mir_datasets/rwc_classical")
    dataset = rwc_classical.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "RWC_C003",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_classical/"),
            "RWC-C/RWC_C003.wav",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_classical/"),
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2/01_annotations_preprocessed/beats/RWC-C/RWC_C003.csv",
        ),
        "midi_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_classical/"),
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2/01_annotations_preprocessed/MIDI_aligned/RWC-C/RWC_C003.mid",
        ),
        "piece_number": "3",
        "cd_number": "1",
        "track_number": "3",
        "title": "Symphony No.5 in C minor, Op.67, 1st mvmt.",
        "artist": "Tokyo City Philharmonic Orchestra",
        "singer_information": "",
        "singing_language": "",
        "tempo": "",
        "variation": "",
        "live_instrument": "Orch",
        "drum_information": "",
        "composer": "Beethoven, Ludwig van",
        "composition_type": "Symphony",
        "main_genre": "Classical",
        "sub_genre": "Classical",
        "audio_start": "0.022",
        "audio_end": "427.6245",
        "duration": "433.7109977324263",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "midi": pretty_midi.PrettyMIDI,
        "notes": annotations.NoteData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)
    # test NoteData
    assert isinstance(track.notes, annotations.NoteData)
    assert len(track.notes.intervals) > 0


def test_load_beats():
    beats_path = (
        "tests/resources/mir_datasets/rwc_classical/"
        + "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2/01_annotations_preprocessed/beats/RWC-C/RWC_C003.csv"
    )
    beat_data = rwc_classical.load_beats(beats_path)

    # check types
    assert type(beat_data) is annotations.BeatData
    assert type(beat_data.times) is np.ndarray
    assert type(beat_data.positions) is np.ndarray

    # check values
    assert np.array_equal(beat_data.times[:4], np.array([0.290, 0.650, 1.650, 2.580]))
    assert np.array_equal(beat_data.positions[:4], np.array([2, 1, 2, 1]))


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/rwc_classical"
    dataset = rwc_classical.Dataset(data_home, version="test")
    metadata = dataset._metadata

    assert "RWC_C003" in metadata
    assert metadata["RWC_C003"]["piece_number"] == "3"
    assert metadata["RWC_C003"]["cd_number"] == "1"
    assert metadata["RWC_C003"]["track_number"] == "3"
    assert metadata["RWC_C003"]["title"] == "Symphony No.5 in C minor, Op.67, 1st mvmt."
    assert metadata["RWC_C003"]["artist"] == "Tokyo City Philharmonic Orchestra"
    assert metadata["RWC_C003"]["live_instrument"] == "Orch"
    assert metadata["RWC_C003"]["main_genre"] == "Classical"
    assert metadata["RWC_C003"]["sub_genre"] == "Classical"
    assert metadata["RWC_C003"]["audio_start"] == "0.022"
    assert metadata["RWC_C003"]["audio_end"] == "427.6245"
    assert metadata["RWC_C003"]["duration"] == "433.7109977324263"


def test_load_metadata_else(tmp_path):
    data_home = tmp_path / "rwc_classical"
    metadata_dir = (
        data_home / "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2"
    )
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "metadata.csv"

    metadata_file.write_text(
        "RWCID;CollID;PieceNo;CDNo;TrackNo;Title;Artist;SingerInformation;"
        "SingingLanguage;Tempo;Variation;LiveInstruments;DrumInformation;"
        "Composer;CompositionType;GenreMain;GenreSub;audio_start;audio_end;duration\n"
        "RWC_C001;J;1;1;1;Test;Test Artist;;;;;;;;;Classical;;0.1;10.0;10.0\n",
        encoding="utf-8",
    )

    dataset = rwc_classical.Dataset(str(data_home), version="test")
    metadata = dataset._metadata

    assert metadata == {}
