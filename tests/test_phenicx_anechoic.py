import os
import shutil
import numpy as np
import pytest

from mirdata.datasets import phenicx_anechoic
from mirdata import annotations, download_utils
from tests.test_utils import run_track_tests, run_multitrack_tests


def test_track():
    default_trackid = "beethoven-violin"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "beethoven-violin",
        "audio_paths": [
            "tests/resources/mir_datasets/PHENICX-Anechoic/"
            + "audio/beethoven/violin1.wav",
            "tests/resources/mir_datasets/PHENICX-Anechoic/"
            + "audio/beethoven/violin2.wav",
            "tests/resources/mir_datasets/PHENICX-Anechoic/"
            + "audio/beethoven/violin3.wav",
            "tests/resources/mir_datasets/PHENICX-Anechoic/"
            + "audio/beethoven/violin4.wav",
        ],
        "notes_path": "tests/resources/mir_datasets/PHENICX-Anechoic/"
        + "annotations/beethoven/violin.txt",
        "notes_original_path": "tests/resources/mir_datasets/PHENICX-Anechoic/"
        + "annotations/beethoven/violin_o.txt",
        "instrument": "violin",
        "piece": "beethoven",
        "n_voices": 4,
    }

    expected_property_types = {
        "notes": annotations.NoteData,
        "notes_original": annotations.NoteData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100,)


def test_get_audio_voice():
    default_trackid = "beethoven-violin"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    track = dataset.track(default_trackid)

    y, sr = track.get_audio_voice(1)
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100,)

    with pytest.raises(AssertionError):
        y, sr = track.get_audio_voice(5)


def test_to_jams():
    default_trackid = "beethoven-violin"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    track = dataset.track(default_trackid)
    jam = track.to_jams()

    assert jam.validate()

    notes = jam.annotations[0]["data"]
    assert [note.time for note in notes] == [4.284082, 4.284082, 4.284082]
    assert [note.duration for note in notes] == [
        0.9872560000000004,
        0.9872560000000004,
        0.9872560000000004,
    ]
    assert [note.value for note in notes] == [
        220.0,
        329.6275569128699,
        554.3652619537442,
    ]


def test_load_score():
    # load a file which exists
    score_path = (
        "tests/resources/mir_datasets/PHENICX-Anechoic/annotations/beethoven/violin.txt"
    )
    note_data = phenicx_anechoic.load_score(score_path)

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.notes) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array([[4.284082, 5.271338], [4.284082, 5.271338], [4.284082, 5.271338]]),
    )
    assert np.allclose(note_data.notes, np.array([220.0, 329.62755691, 554.36526195]))


def test_multitrack():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)
    # import pdb;pdb.set_trace()
    expected_attributes = {
        "mtrack_id": "beethoven",
        "track_audio_property": "audio",
        "track_ids": [
            "beethoven-horn",
            "beethoven-doublebass",
            "beethoven-violin",
            "beethoven-bassoon",
            "beethoven-flute",
            "beethoven-clarinet",
            "beethoven-viola",
            "beethoven-oboe",
            "beethoven-cello",
            "beethoven-trumpet",
        ],
        "instruments": {
            "horn": "beethoven-horn",
            "doublebass": "beethoven-doublebass",
            "violin": "beethoven-violin",
            "bassoon": "beethoven-bassoon",
            "flute": "beethoven-flute",
            "clarinet": "beethoven-clarinet",
            "viola": "beethoven-viola",
            "oboe": "beethoven-oboe",
            "cello": "beethoven-cello",
            "trumpet": "beethoven-trumpet",
        },
        "sections": {
            "brass": ["beethoven-horn", "beethoven-trumpet"],
            "strings": [
                "beethoven-doublebass",
                "beethoven-violin",
                "beethoven-viola",
                "beethoven-cello",
            ],
            "woodwinds": [
                "beethoven-bassoon",
                "beethoven-flute",
                "beethoven-clarinet",
                "beethoven-oboe",
            ],
        },
        "piece": "beethoven",
    }

    expected_property_types = {
        "tracks": dict,
        "track_audio_property": str,
    }

    run_track_tests(mtrack, expected_attributes, expected_property_types)
    run_multitrack_tests(mtrack)


def test_get_audio_for_instrument():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    y = mtrack.get_audio_for_instrument("violin")
    assert y.shape == (44100,)

    with pytest.raises(AssertionError):
        y = mtrack.get_audio_for_instrument("guitar")


def test_get_audio_for_section():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    y = mtrack.get_audio_for_section("strings")
    assert y.shape == (1, 44100)

    with pytest.raises(AssertionError):
        y = mtrack.get_audio_for_section("synths")


def test_get_notes_target():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    track_keys = ["beethoven-viola", "beethoven-violin"]
    note_data = mtrack.get_notes_target(track_keys, notes_property="notes")

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.notes) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array(
            [
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.310204, 4.910204],
                [4.310204, 4.910204],
                [8.359184, 12.004082],
            ]
        ),
    )
    assert np.allclose(
        note_data.notes,
        np.array([220.0, 329.62755691, 554.36526195, 220.0, 329.62755691, 220.0]),
    )


def test_get_notes_for_instrument():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    note_data = mtrack.get_notes_for_instrument(
        instrument="violin", notes_property="notes"
    )
    # import pdb;pdb.set_trace()

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.notes) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array([[4.284082, 5.271338], [4.284082, 5.271338], [4.284082, 5.271338]]),
    )
    assert np.allclose(note_data.notes, np.array([220.0, 329.62755691, 554.36526195]))


def test_get_notes_for_section():
    default_trackid = "beethoven"
    data_home = "tests/resources/mir_datasets/PHENICX-Anechoic"
    dataset = phenicx_anechoic.Dataset(data_home)
    mtrack = dataset.multitrack(default_trackid)

    note_data = mtrack.get_notes_for_section(section="strings", notes_property="notes")

    # check types
    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) is np.ndarray
    assert type(note_data.notes) is np.ndarray

    # check values
    assert np.array_equal(
        note_data.intervals,
        np.array(
            [
                [4.260862, 6.780091],
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.284082, 5.271338],
                [4.310204, 4.910204],
                [4.310204, 4.910204],
                [4.331995, 6.621655],
                [8.359184, 12.004082],
                [12.167256, 14.038594],
                [12.213696, 13.862268],
                [19.783401, 21.656599],
                [19.841451, 21.462971],
            ]
        ),
    )
    assert np.allclose(
        note_data.notes,
        np.array(
            [
                55.0,
                220.0,
                329.62755691,
                554.36526195,
                220.0,
                329.62755691,
                110.0,
                220.0,
                51.9130872,
                103.82617439,
                48.9994295,
                97.998859,
            ]
        ),
    )


# def test_track():
#     default_trackid = 'beethoven'
#     data_home = 'tests/resources/mir_datasets/PHENICX-Anechoic'
#     instruments = [
#         'bassoon',
#         'cello',
#         'clarinet',
#         'doublebass',
#         'flute',
#         'horn',
#         'oboe',
#         'trumpet',
#         'viola',
#         'violin',
#     ]
#     sections = ['brass', 'strings', 'woodwinds']
#     no_sources_per_instrument = [2, 1, 2, 1, 2, 2, 2, 2, 2, 4]
#     section_id = [2, 1, 2, 1, 2, 0, 2, 0, 1, 1]
#     all_sources = []
#     sources4sections = [[] for section in sections]
#     targets = collections.OrderedDict()
#     sources = collections.OrderedDict()
#     i = 0
#     for noinst, (idi, instrument) in zip(
#         no_sources_per_instrument, enumerate(instruments)
#     ):
#         temp_sources = []
#         if noinst > 1:
#             for n in range(noinst):
#                 source = phenicx_anechoic.Source(
#                     name=instrument + str(n + 1),
#                     stem_id=str(i),
#                     path=os.path.join(
#                         data_home,
#                         'audio',
#                         'beethoven',
#                         instrument + str(n + 1) + '.wav',
#                     ),
#                 )
#                 temp_sources.append(source)
#                 sources[instrument + str(n + 1)] = source
#                 sources4sections[section_id[idi]].append(source)
#                 i += 1
#         else:
#             source = phenicx_anechoic.Source(
#                 name=instrument,
#                 stem_id=str(i),
#                 path=os.path.join(data_home, 'audio', 'beethoven', instrument + '.wav'),
#             )
#             temp_sources.append(source)
#             sources[instrument] = source
#             sources4sections[section_id[idi]].append(source)
#             i += 1
#         all_sources.extend(temp_sources)

#         targets[instrument] = phenicx_anechoic.Target(
#             sources=temp_sources,
#             name=instrument,
#             instruments=[instrument],
#             score_path=os.path.join(data_home, 'annotations', 'beethoven'),
#         )

#     for sid, section in enumerate(sections):
#         section_inst = [
#             instrument
#             for idi, instrument in enumerate(instruments)
#             if section_id[idi] == sid
#         ]
#         targets[section] = phenicx_anechoic.Target(
#             sources=sources4sections[sid],
#             name=section,
#             instruments=section_inst,
#             score_path=os.path.join(data_home, 'annotations', 'beethoven'),
#         )

#     mix = phenicx_anechoic.Target(
#         sources=all_sources,
#         name='mix',
#         instruments=instruments,
#         score_path=os.path.join(data_home, 'annotations', 'beethoven'),
#     )

#     track = phenicx_anechoic.Track(default_trackid, data_home=data_home)
#     expected_attributes = {
#         'track_id': 'beethoven',
#         'annotation_path': "tests/resources/mir_datasets/PHENICX-Anechoic/annotations/beethoven",
#         'audio_path': "tests/resources/mir_datasets/PHENICX-Anechoic/audio/beethoven",
#         'instruments': instruments,
#         'sections': sections,
#         'mix': mix,
#         'targets': targets,
#         'sources': sources,
#     }

#     expected_property_types = {
#         'instruments': list,
#         'sections': list,
#         'mix': phenicx_anechoic.Target,
#         'sources': collections.OrderedDict,
#         'targets': collections.OrderedDict,
#     }

#     run_track_tests(track, expected_attributes, expected_property_types)

#     y = track.mix.audio
#     z, sr = track.get_audio_mix()
#     # import pdb;pdb.set_trace()
#     assert np.allclose(y, z)
#     assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
#     assert y.shape == (44100,)

#     for instrument in instruments:
#         y = track.targets[instrument].audio
#         z, sr = track.get_audio_target(instrument)
#         assert np.allclose(y, z)
#         assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
#         assert y.shape == (44100,)

#     for instrument in sections:
#         y = track.targets[section].audio
#         z, sr = track.get_audio_target(section)
#         assert np.allclose(y, z)
#         assert sr == 44100, 'sample rate of PHENICX-Anechoic should be 44100'
#         assert y.shape == (44100,)

# def test_track():
#     default_trackid = "Beethoven-S3-I-ex1"
#     data_home = "tests/resources/mir_datasets/orchset"
#     dataset = orchset.Dataset(data_home)
#     track = dataset.track(default_trackid)

#     expected_attributes = {
#         "track_id": "Beethoven-S3-I-ex1",
#         "audio_path_mono": "tests/resources/mir_datasets/orchset/"
#         + "audio/mono/Beethoven-S3-I-ex1.wav",
#         "audio_path_stereo": "tests/resources/mir_datasets/orchset/"
#         + "audio/stereo/Beethoven-S3-I-ex1.wav",
#         "melody_path": "tests/resources/mir_datasets/orchset/"
#         + "GT/Beethoven-S3-I-ex1.mel",
#         "composer": "Beethoven",
#         "work": "S3-I",
#         "excerpt": "1",
#         "predominant_melodic_instruments": ["strings", "winds"],
#         "alternating_melody": True,
#         "contains_winds": True,
#         "contains_strings": True,
#         "contains_brass": False,
#         "only_strings": False,
#         "only_winds": False,
#         "only_brass": False,
#     }

#     expected_property_types = {
#         "melody": annotations.F0Data,
#         "audio_mono": tuple,
#         "audio_stereo": tuple,
#     }

#     run_track_tests(track, expected_attributes, expected_property_types)

#     y_mono, sr_mono = track.audio_mono
#     assert sr_mono == 44100
#     assert y_mono.shape == (44100 * 2,)

#     y_stereo, sr_stereo = track.audio_stereo
#     assert sr_stereo == 44100
#     assert y_stereo.shape == (2, 44100 * 2)


# def test_to_jams():

#     data_home = "tests/resources/mir_datasets/orchset"
#     dataset = orchset.Dataset(data_home)
#     track = dataset.track("Beethoven-S3-I-ex1")
#     jam = track.to_jams()

#     f0s = jam.search(namespace="pitch_contour")[0]["data"]
#     assert [f0.time for f0 in f0s] == [0.0, 0.08, 0.09]
#     assert [f0.duration for f0 in f0s] == [0.0, 0.0, 0.0]
#     assert [f0.value for f0 in f0s] == [
#         {"frequency": 0.0, "index": 0, "voiced": False},
#         {"frequency": 0.0, "index": 0, "voiced": False},
#         {"frequency": 622.254, "index": 0, "voiced": True},
#     ]
#     assert [f0.confidence for f0 in f0s] == [0.0, 0.0, 1.0]

#     assert jam["sandbox"]["alternating_melody"] == True


# def test_load_melody():
#     # load a file which exists
#     melody_path = "tests/resources/mir_datasets/orchset/GT/Beethoven-S3-I-ex1.mel"
#     melody_data = orchset.load_melody(melody_path)

#     # check types
#     assert type(melody_data) == annotations.F0Data
#     assert type(melody_data.times) is np.ndarray
#     assert type(melody_data.frequencies) is np.ndarray
#     assert type(melody_data.confidence) is np.ndarray

#     # check values
#     assert np.array_equal(melody_data.times, np.array([0.0, 0.08, 0.09]))
#     assert np.array_equal(melody_data.frequencies, np.array([0.0, 0.0, 622.254]))
#     assert np.array_equal(melody_data.confidence, np.array([0.0, 0.0, 1.0]))


# def test_load_metadata():
#     data_home = "tests/resources/mir_datasets/orchset"
#     dataset = orchset.Dataset(data_home)
#     metadata = dataset._metadata
#     assert metadata["Beethoven-S3-I-ex1"] == {
#         "predominant_melodic_instruments-raw": "strings+winds",
#         "predominant_melodic_instruments-normalized": ["strings", "winds"],
#         "alternating_melody": True,
#         "contains_winds": True,
#         "contains_strings": True,
#         "contains_brass": False,
#         "only_strings": False,
#         "only_winds": False,
#         "only_brass": False,
#         "composer": "Beethoven",
#         "work": "S3-I",
#         "excerpt": "1",
#     }
#     assert metadata["Haydn-S94-Menuet-ex1"] == {
#         "predominant_melodic_instruments-raw": "string+winds",
#         "predominant_melodic_instruments-normalized": ["strings", "winds"],
#         "alternating_melody": True,
#         "contains_winds": True,
#         "contains_strings": True,
#         "contains_brass": False,
#         "only_strings": False,
#         "only_winds": False,
#         "only_brass": False,
#         "composer": "Haydn",
#         "work": "S94-Menuet",
#         "excerpt": "1",
#     }
#     assert metadata["Musorgski-Ravel-PicturesExhibition-Promenade1-ex2"] == {
#         "predominant_melodic_instruments-raw": "strings",
#         "predominant_melodic_instruments-normalized": ["strings"],
#         "alternating_melody": False,
#         "contains_winds": True,
#         "contains_strings": False,
#         "contains_brass": False,
#         "only_strings": True,
#         "only_winds": False,
#         "only_brass": False,
#         "composer": "Musorgski-Ravel",
#         "work": "PicturesExhibition-Promenade1",
#         "excerpt": "2",
#     }
#     assert metadata["Rimski-Korsakov-Scheherazade-YoungPrincePrincess-ex4"] == {
#         "predominant_melodic_instruments-raw": "strings+winds",
#         "predominant_melodic_instruments-normalized": ["strings", "winds"],
#         "alternating_melody": True,
#         "contains_winds": True,
#         "contains_strings": True,
#         "contains_brass": False,
#         "only_strings": False,
#         "only_winds": False,
#         "only_brass": False,
#         "composer": "Rimski-Korsakov",
#         "work": "Scheherazade-YoungPrincePrincess",
#         "excerpt": "4",
#     }
#     assert metadata["Schubert-S8-II-ex2"] == {
#         "predominant_melodic_instruments-raw": "winds (solo)",
#         "predominant_melodic_instruments-normalized": ["winds"],
#         "alternating_melody": False,
#         "contains_winds": False,
#         "contains_strings": True,
#         "contains_brass": False,
#         "only_strings": False,
#         "only_winds": True,
#         "only_brass": False,
#         "composer": "Schubert",
#         "work": "S8-II",
#         "excerpt": "2",
#     }


# def test_download(httpserver):
#     data_home = "tests/resources/mir_datasets/orchset_download"
#     if os.path.exists(data_home):
#         shutil.rmtree(data_home)

#     httpserver.serve_content(
#         open("tests/resources/download/Orchset_dataset_0.zip", "rb").read()
#     )

#     remotes = {
#         "all": download_utils.RemoteFileMetadata(
#             filename="Orchset_dataset_0.zip",
#             url=httpserver.url,
#             checksum=("4794bc3514f7e8d1727f0d975d6d1ee2"),
#             unpack_directories=["Orchset"],
#         )
#     }
#     dataset = orchset.Dataset(data_home)
#     dataset.remotes = remotes
#     dataset.download(None, False, False)

#     assert os.path.exists(data_home)
#     assert not os.path.exists(os.path.join(data_home, "Orchset"))

#     assert os.path.exists(os.path.join(data_home, "README.txt"))
#     assert os.path.exists(
#         os.path.join(data_home, "Orchset - Predominant Melodic Instruments.csv")
#     )
#     track = dataset.track("Beethoven-S3-I-ex1")
#     assert os.path.exists(track.audio_path_mono)
#     assert os.path.exists(track.audio_path_stereo)
#     assert os.path.exists(track.melody_path)

#     # test downloading again
#     dataset.download(None, False, False)

#     if os.path.exists(data_home):
#         shutil.rmtree(data_home)

#     # test downloading twice with cleanup
#     dataset.download(None, False, True)
#     dataset.download(None, False, False)

#     if os.path.exists(data_home):
#         shutil.rmtree(data_home)

#     # test downloading twice with force overwrite
#     dataset.download(None, False, False)
#     dataset.download(None, True, False)

#     if os.path.exists(data_home):
#         shutil.rmtree(data_home)

#     # test downloading twice with force overwrite and cleanup
#     dataset.download(None, False, True)
#     dataset.download(None, True, False)

#     if os.path.exists(data_home):
#         shutil.rmtree(data_home)
