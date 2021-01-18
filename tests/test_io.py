import tempfile
from io import BufferedReader, BytesIO, StringIO, TextIOWrapper

import pytest

from mirdata import io


def test_coerce_to_string_with_none():
    @io.coerce_to_string_io
    def func(fh):
        assert fh is None

    func(None)


def test_coerce_to_string_io_with_path():
    with tempfile.NamedTemporaryFile() as f:

        @io.coerce_to_string_io
        def func(fh):
            assert isinstance(fh, TextIOWrapper)

        func(f.name)


def test_coerce_to_string_io_with_stringio():
    @io.coerce_to_string_io
    def func(fh):
        assert isinstance(fh, StringIO)

    with StringIO("abc") as f:
        func(f)


def test_invalid_coerce_to_string_io():
    @io.coerce_to_string_io
    def func(fh):
        raise RuntimeError("YOU SHOULDNT BE HERE")

    with pytest.raises(ValueError):
        func(123)


def test_coerce_to_bytes_with_none():
    @io.coerce_to_bytes_io
    def func(fh):
        assert fh is None

    func(None)


def test_coerce_to_bytes_io_with_path():
    with tempfile.NamedTemporaryFile() as f:

        @io.coerce_to_bytes_io
        def func(fh):
            assert isinstance(fh, BufferedReader)

        func(f.name)


def test_coerce_to_bytes_io_with_bytesio():
    @io.coerce_to_bytes_io
    def func(fh):
        assert isinstance(fh, BytesIO)

    with BytesIO(b"abc") as f:
        func(f)


def test_invalid_coerce_to_bytes_io():
    @io.coerce_to_bytes_io
    def func(fh):
        raise RuntimeError("YOU SHOULDNT BE HERE")

    with pytest.raises(ValueError):
        func(123)
