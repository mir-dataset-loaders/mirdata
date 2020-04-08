# -*- coding: utf-8 -*-

import sys
import pytest

from mirdata import track

if sys.version_info.major == 3:
    builtin_module_name = 'builtins'
else:
    builtin_module_name = '__builtin__'


def test_track_repr():
    class TestTrack(track.Track):
        def __init__(self):
            self.a = 'asdf'
            self.b = 1.2345678
            self.c = {1: 'a', 'b': 2}
            self._d = 'hidden'
            self.e = None
            self.long = 'a' + 'b' * 50 + 'c' * 50

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
    expected3 = """long="...{}",\n  """.format('b' * 50 + 'c' * 50)
    expected4 = """f: ThisObjectType,\n  g: I have an improper docstring,\n)"""

    test_track = TestTrack()
    actual = test_track.__repr__()
    assert actual == expected1 + expected2 + expected3 + expected4

    with pytest.raises(NotImplementedError):
        test_track.to_jams()

    class NoDocsTrack(track.Track):
        @property
        def no_doc(self):
            return "whee!"

    bad_track = NoDocsTrack()
    with pytest.raises(ValueError):
        bad_track.__repr__()
