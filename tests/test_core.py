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


def test_multitrack_repr():
    class TestTrack(core.Track):
        def __init__(self):
            self.a = "asdf"

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

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
    expected2 = """c={1: \'a\', \'b\': 2},\n  e=None,\n  """
    expected3 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected4 = """mtrack_id="test",\n  track_ids=[\'a\', \'b\', \'c\'],\n  """
    expected5 = """f: ThisObjectType,\n  g: I have an improper docstring,\n  """
    expected6 = """track_audio_property: ,\n  tracks: ,\n)"""

    test_mtrack = TestMultiTrack("test", "foo")
    actual = test_mtrack.__repr__()
    assert (
        actual == expected1 + expected2 + expected3 + expected4 + expected5 + expected6
    )

    with pytest.raises(NotImplementedError):
        test_mtrack.to_jams()


def test_dataset():
    dataset = mirdata.initialize("guitarset")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("rwc_jazz")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("ikala")
    assert isinstance(dataset, core.Dataset)

    dataset = mirdata.initialize("phenicx_anechoic")
    assert isinstance(dataset, core.Dataset)

    print(dataset)  # test that repr doesn't fail


def test_dataset_errors():
    with pytest.raises(ValueError):
        mirdata.initialize("not_a_dataset")

    d = mirdata.initialize("orchset")
    d._track_class = None
    with pytest.raises(NotImplementedError):
        d.track("asdf")

    with pytest.raises(NotImplementedError):
        d.load_tracks()

    with pytest.raises(KeyError):
        d.load_multitracks()

    with pytest.raises(NotImplementedError):
        d.choice_track()

    d = mirdata.initialize("acousticbrainz_genre")
    with pytest.raises(FileNotFoundError):
        d._index


def test_multitrack_basic():
    class TestTrack(core.Track):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key
            self._metadata = {1: "a", "b": 2}

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self.track_ids = []

    mtrack = TestMultiTrack1("test", "foo")

    with pytest.raises(NotImplementedError):
        mtrack.to_jams()

    with pytest.raises(KeyError):
        mtrack.get_target(["a"])

    with pytest.raises(AssertionError):
        mtrack.get_random_target()

    with pytest.raises(AssertionError):
        mtrack.get_mix()

    with pytest.raises(AttributeError):
        mtrack._multitrack_medata

    class TestMultiTrack2(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]
            self._metadata = {1: "a", "b": 2}

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            #### the attribute of Track which returns the relevant audio file for mixing
            return "f"

    mtrack = TestMultiTrack2("test", "foo")
    mtrack.to_jams()
    mtrack.get_target(["a"])
    mtrack.get_random_target()


def test_multitrack_mixing():
    class TestTrack(core.Track):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

        @property
        def track_audio_property(self):
            #### the attribute of Track which returns the relevant audio file for mixing
            return "f"

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
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, np.random.randint(50, 100))), 1000

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

        @property
        def track_audio_property(self):
            #### the attribute of Track which returns the relevant audio file for mixing
            return "f"

    mtrack = TestMultiTrack("test", "foo")

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])

    with pytest.raises(KeyError):
        mtrack.get_target(["d", "e"])

    target1 = mtrack.get_target(["a", "b", "c"], enforce_length=False)
    assert target1.shape[0] == 2
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["a", "b", "c"], average=False, enforce_length=False)
    assert target2.shape[0] == 2
    assert np.max(np.abs(target2)) <= 3


def test_multitrack_unequal_sr():
    class TestTrack(core.Track):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), np.random.randint(10, 1000)

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

        @property
        def track_audio_property(self):
            #### the attribute of Track which returns the relevant audio file for mixing
            return "f"

    mtrack = TestMultiTrack("test", "foo")

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])


def test_multitrack_mono():
    ### no first channel - audio shapes (100,)
    class TestTrack(core.Track):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (100)), 1000

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

        @property
        def track_audio_property(self):
            #### the attribute of Track which returns the relevant audio file for mixing
            return "f"

    mtrack = TestMultiTrack("test", "foo")

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2

    ### one channel mono shape (1, 100)
    class TestTrack1(core.Track):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (1, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

        @property
        def track_audio_property(self):
            #### the attribute of Track which returns the relevant audio file for mixing
            return "f"

    mtrack = TestMultiTrack1("test", "foo")

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2
