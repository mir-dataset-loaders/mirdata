# -*- coding: utf-8 -*-

import pytest
import numpy as np

import mirdata
from mirdata import core


def test_track_repr():
    class TestTrack(core.Track):
        def __init__(self):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50

        @property
        def f(self):
            """ThisObjectType: I have a docstring"""
            return None

        @property
        def g(self):
            """I have an improper docstring"""
            return None

        def h(self):
            return "I'm a function!"

    expected1 = """Track(\n  a="asdf",\n  b=1.2345678,\n  """
    expected2 = """c={1: 'a', 'b': 2},\n  e=None,\n  """
    expected3 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected4 = """f: ThisObjectType,\n  g: I have an improper docstring,\n)"""

    test_track = TestTrack()
    actual = test_track.__repr__()
    assert actual == expected1 + expected2 + expected3 + expected4

    with pytest.raises(NotImplementedError):
        test_track.to_jams()


def test_dataset():
    dataset = mirdata.initialize("guitarset")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("rwc_jazz")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("ikala")
    assert isinstance(dataset, core.Dataset)

    print(dataset)  # test that repr doesn't fail


def test_dataset_errors():
    with pytest.raises(ValueError):
        mirdata.initialize("not_a_dataset")

    d = mirdata.initialize("orchset")
    d._track_object = None
    with pytest.raises(NotImplementedError):
        d.track("asdf")

    with pytest.raises(NotImplementedError):
        d.load_tracks()

    with pytest.raises(NotImplementedError):
        d.choice_track()


def test_multitrack_basic():
    class TestTrack(core.Track):
        def __init__(self, key):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home

    mtrack = TestMultiTrack1("test", "foo")

    with pytest.raises(NotImplementedError):
        mtrack.to_jams()

    with pytest.raises(NotImplementedError):
        mtrack.get_target(["a"])

    with pytest.raises(NotImplementedError):
        mtrack.get_random_target()

    with pytest.raises(NotImplementedError):
        mtrack.get_mix()

    class TestMultiTrack2(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self.tracks = {t: TestTrack(t) for t in ["a", "b", "c"]}
            self.track_audio_property = "f"

        def to_jams(self):
            return None

    mtrack = TestMultiTrack2("test", "foo")
    mtrack.to_jams()
    mtrack.get_target(["a"])
    mtrack.get_random_target()
    mtrack.get_mix()


def test_multitrack_mixing():
    class TestTrack(core.Track):
        def __init__(self, key):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self.tracks = {t: TestTrack(t) for t in ["a", "b", "c"]}
            self.track_audio_property = "f"

    mtrack = TestMultiTrack("test", "foo")

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (2, 100)
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["b", "c"], weights=[0.5, 0.2])
    assert target2.shape == (2, 100)
    assert np.max(np.abs(target2)) <= 1

    target3 = mtrack.get_target(["b", "c"], weights=[0.5, 5])
    assert target3.shape == (2, 100)
    assert np.max(np.abs(target3)) <= 1

    target4 = mtrack.get_target(["a", "c"], average=False)
    assert target4.shape == (2, 100)
    assert np.max(np.abs(target4)) <= 2

    target5 = mtrack.get_target(["a", "c"], average=False, weights=[0.1, 0.5])
    assert target5.shape == (2, 100)
    assert np.max(np.abs(target5)) <= 0.6

    random_target1, t1, w1 = mtrack.get_random_target(n_tracks=2)
    assert random_target1.shape == (2, 100)
    assert np.max(np.abs(random_target1)) <= 1
    assert len(t1) == 2
    assert len(w1) == 2
    assert np.all(w1 >= 0.3)
    assert np.all(w1 <= 1.0)

    random_target2, t2, w2 = mtrack.get_random_target(n_tracks=5)
    assert random_target2.shape == (2, 100)
    assert np.max(np.abs(random_target2)) <= 1
    assert len(t2) == 3
    assert len(w2) == 3
    assert np.all(w2 >= 0.3)
    assert np.all(w2 <= 1.0)

    random_target3, t3, w3 = mtrack.get_random_target()
    assert random_target3.shape == (2, 100)
    assert np.max(np.abs(random_target3)) <= 1
    assert len(t3) == 3
    assert len(w3) == 3
    assert np.all(w3 >= 0.3)
    assert np.all(w3 <= 1.0)

    random_target4, t4, w4 = mtrack.get_random_target(
        n_tracks=2, min_weight=0.1, max_weight=0.4
    )
    assert random_target4.shape == (2, 100)
    assert np.max(np.abs(random_target4)) <= 1
    assert len(t4) == 2
    assert len(w4) == 2
    assert np.all(w4 >= 0.1)
    assert np.all(w4 <= 0.4)

    mix = mtrack.get_mix()
    assert mix.shape == (2, 100)


def test_multitrack_unequal_len():
    class TestTrack(core.Track):
        def __init__(self, key):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, np.random.randint(50, 100))), 1000

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self.tracks = {t: TestTrack(t) for t in ["a", "b", "c"]}
            self.track_audio_property = "f"

    mtrack = TestMultiTrack("test", "foo")

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])

    target1 = mtrack.get_target(["a", "b", "c"], enforce_length=False)
    assert target1.shape[0] == 2
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["a", "b", "c"], average=False, enforce_length=False)
    assert target2.shape[0] == 2
    assert np.max(np.abs(target2)) <= 3


def test_multitrack_unequal_sr():
    class TestTrack(core.Track):
        def __init__(self, key):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), np.random.randint(10, 1000)

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self.tracks = {t: TestTrack(t) for t in ["a", "b", "c"]}
            self.track_audio_property = "f"

    mtrack = TestMultiTrack("test", "foo")

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])


def test_multitrack_mono():
    ### no first channel - audio shapes (100,)
    class TestTrack(core.Track):
        def __init__(self, key):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (100)), 1000

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self.tracks = {t: TestTrack(t) for t in ["a", "b", "c"]}
            self.track_audio_property = "f"

    mtrack = TestMultiTrack("test", "foo")

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2

    ### one channel mono shape (1, 100)
    class TestTrack1(core.Track):
        def __init__(self, key):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (1, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self.tracks = {t: TestTrack1(t) for t in ["a", "b", "c"]}
            self.track_audio_property = "f"

    mtrack = TestMultiTrack1("test", "foo")

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2
