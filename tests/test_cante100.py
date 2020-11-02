# -*- coding: utf-8 -*-

import os
import numpy as np

from tests.test_utils import run_track_tests

from mirdata import cante100, utils
from tests.test_utils import DEFAULT_DATA_HOME

TEST_DATA_HOME = "tests/resources/mir_datasets/cante100"


def test_track_default_data_home():
    # test data home None
    track_default = cante100.Track("008")
    assert track_default._data_home == os.path.join(DEFAULT_DATA_HOME, "cante100")


def test_track():
    default_trackid = "008"
    track = cante100.Track(default_trackid, data_home=TEST_DATA_HOME)
    expected_attributes = {
        'artist': 'Toronjo',
        'duration': 179.0,
        'audio_path': 'tests/resources/mir_datasets/cante100/cante100audio/008_PacoToronjo_'
        + 'Fandangos.mp3',
        'f0_path': 'tests/resources/mir_datasets/cante100/cante100midi_f0/008_PacoToronjo_'
        + 'Fandangos.f0.csv',
        'identifier': '4eebe839-82bb-426e-914d-7c4525dd9dad',
        'notes_path': 'tests/resources/mir_datasets/cante100/cante100_automaticTranscription/008_PacoToronjo_'
        + 'Fandangos.notes.csv',
        'release': 'Atlas del cante flamenco',
        'spectrum_path': 'tests/resources/mir_datasets/cante100/cante100_spectrum/008_PacoToronjo_'
        + 'Fandangos.spectrum.csv',
        'title': 'Huelva Como Capital',
        'track_id': '008',
    }

    expected_property_types = {
        'melody': utils.F0Data,
        'notes': utils.NoteData
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 22050
    assert audio.shape == (2, 3956358)


def test_to_jams():
    default_trackid = "008"
    track = cante100.Track(default_trackid, data_home=TEST_DATA_HOME)
    jam = track.to_jams()

    # Validate cante100 jam schema
    assert jam.validate()

    notes = jam.search(namespace='note_hz')[0]['data']
    melody = jam.search(namespace='pitch_contour')[0]['data']

    assert [note.time for note in notes] == [25.7625, 26.1457, 37.3319, 37.5612, 37.7876, 44.8755, 45.1686,
                                             46.0219, 46.1961, 46.5589, 46.884, 47.2149, 50.5673, 50.7269,
                                             51.2174, 51.3916, 51.557, 51.7225, 52.0185, 52.1839, 53.4494,
                                             53.8558, 54.0125, 54.1751, 54.3405, 54.503, 54.6598, 54.9529,
                                             55.6176, 55.9107, 56.3229, 56.5232, 56.7496, 56.8483, 60.8856,
                                             61.1701, 68.7833, 69.1316, 69.448, 69.8253, 70.0227, 70.1823,
                                             70.3129, 70.609, 70.9137, 71.085, 71.2301, 71.3694, 71.6568,
                                             71.9615, 72.1444, 72.2953, 72.4753, 72.7307, 72.8642, 74.3155,
                                             74.5099, 74.8176, 75.0411, 75.2472, 75.3575, 75.5258, 75.7116,
                                             76.0744, 76.5881, 76.6984, 77.0206, 77.1657, 77.4792, 77.6475,
                                             77.7897, 79.3919, 79.6909, 79.836, 80.2453, 80.5152, 80.701,
                                             80.8838, 81.3395, 81.6356, 81.8068, 81.9664, 82.1261, 82.4076,
                                             84.6977, 84.898, 85.0605, 85.255, 85.7571, 86.0677, 86.7962,
                                             87.4464, 87.7569, 87.9108, 89.6377, 89.838, 90.0963, 90.7639,
                                             90.9816, 91.1354, 91.3125, 91.6259, 91.7885, 91.9452, 92.0961,
                                             92.3893, 92.6999, 92.8798, 93.3036, 93.501, 93.8434, 94.7984,
                                             95.4195, 95.7881, 96.0784, 96.6095, 97.0159, 97.5964, 97.8315,
                                             98.3249, 98.499, 98.6674, 98.8328, 98.9983, 99.1608, 99.6658,
                                             100.423, 100.824, 100.986, 101.155, 101.323, 101.486, 101.64,
                                             101.953, 102.266, 102.458, 102.612, 102.792, 102.969, 103.245,
                                             103.509, 103.909, 134.748, 134.908, 135.219, 135.488, 135.874,
                                             136.057, 136.496, 136.777, 136.928, 137.079, 137.23, 137.514,
                                             140.234, 140.768, 141.247, 141.523, 142.512, 142.986, 143.154,
                                             143.319, 145.554, 145.975, 146.262, 146.547, 146.779, 147.22,
                                             147.49, 147.641, 147.795, 147.943, 148.225, 148.524, 150.66,
                                             150.825, 151.179, 151.362, 151.824, 152.027, 152.732, 153.046,
                                             153.356, 153.507, 155.664, 156.009, 156.302, 156.601, 156.659,
                                             157.455, 157.745, 157.983, 158.258, 158.412, 158.569, 158.72,
                                             159.007, 159.318, 159.501, 160.049, 161.001, 161.582, 161.823,
                                             162.142, 162.496, 162.842, 163.318, 163.805, 164.008, 164.484,
                                             164.65, 164.812, 164.978, 165.291, 165.46, 166.734, 167.126,
                                             167.297, 167.48, 167.66, 167.825, 167.979, 168.292, 168.594,
                                             168.858, 169.053, 169.236, 169.392, 169.781, 170.179, 170.562]
    assert notes[0].duration == 0.3453969999999984
    assert notes[-1].duration == 0.3976419999999905
    assert notes[0].value == 207.65234878997256
    assert notes[-1].value == 207.65234878997256

    assert melody[0].time == 0.023219954
    assert melody[-1].time == 179.795011337
    assert melody[0].value == {'index': 0, 'frequency': 0, 'voiced': False}
    assert melody[-1].value == {'index': 0, 'frequency': -110.0, 'voiced': False}


def test_load_f0():
    # load a file which exists
    f0_path = 'tests/resources/mir_datasets/cante100/cante100midi_f0/008_PacoToronjo_Fandangos.f0.csv'
    f0_data = cante100.load_melody(f0_path)

    # check types
    assert type(f0_data) == utils.F0Data
    assert type(f0_data.times) is np.ndarray
    assert type(f0_data.frequencies) is np.ndarray
    assert type(f0_data.confidence) is np.ndarray


def test_load_metadata():
    data_home = 'tests/resources/mir_datasets/cante100'
    metadata = cante100._load_metadata(data_home)
    assert metadata['data_home'] == data_home
    assert metadata['008'] == {
        'musicBrainzID': '4eebe839-82bb-426e-914d-7c4525dd9dad',
        'artist': 'Toronjo',
        'title': 'Huelva Como Capital',
        'release': 'Atlas del cante flamenco',
        'duration': 179
    }
