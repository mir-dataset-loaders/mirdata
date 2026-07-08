"""Tests for the Vienna 4x22 loader."""

import logging
import os

try:
    import partitura
except ImportError:
    logging.error(
        "In order to test vienna4x22 you must have partitura installed. "
        "Please reinstall mirdata using `pip install 'mirdata[vienna4x22]'` "
        "and re-run the tests."
    )
    raise ImportError

import numpy as np

from mirdata.datasets import vienna4x22
from tests.test_utils import run_track_tests

DATA_HOME = os.path.normpath("tests/resources/mir_datasets/vienna4x22")
TRACK_ID = "Chopin_op10_no3_p01"


def test_track():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    expected_attributes = {
        "track_id": TRACK_ID,
        "piece": "Chopin_op10_no3",
        "pianist_id": "01",
        "score_path": os.path.join(DATA_HOME, "musicxml/Chopin_op10_no3.musicxml"),
        "performance_path": os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid"),
        "match_path": os.path.join(DATA_HOME, "match/Chopin_op10_no3_p01.match"),
    }

    expected_property_types = {
        "score": partitura.score.Score,
        "performance": partitura.performance.Performance,
        "match": tuple,
        "note_array": np.ndarray,
        "performance_note_array": np.ndarray,
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_load_score():
    path = os.path.join(DATA_HOME, "musicxml/Chopin_op10_no3.musicxml")
    score = vienna4x22.load_score(path)
    assert isinstance(score, partitura.score.Score)
    assert vienna4x22.load_score(None) is None


def test_load_performance():
    path = os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid")
    perf = vienna4x22.load_performance(path)
    assert isinstance(perf, partitura.performance.Performance)
    na = perf.note_array()
    assert na.shape[0] > 0
    assert vienna4x22.load_performance(None) is None


def test_load_match():
    path = os.path.join(DATA_HOME, "match/Chopin_op10_no3_p01.match")
    result = vienna4x22.load_match(path)
    assert isinstance(result, tuple)
    assert len(result) == 3
    performance, alignment, score = result
    assert isinstance(performance, partitura.performance.Performance)
    assert isinstance(alignment, list) and len(alignment) > 0
    assert isinstance(score, partitura.score.Score)
    assert vienna4x22.load_match(None) is None
