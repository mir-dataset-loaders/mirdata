"""Tests for MULTIVOX dataset"""

import os
import wave
from unittest.mock import patch

import pytest

from mirdata.datasets import multivox
from tests.test_utils import run_track_tests


# Helper to construct a Track with minimal index/metadata (used by unit tests)
def _make_track(tmp_path, meta, paths):
    track_id = "tid"
    index = {"version": "test", "tracks": {track_id: paths}}

    def metadata():
        return {track_id: meta}

    return multivox.Track(track_id, str(tmp_path), "multivox", index, metadata)


def test_track():
    default_trackid = "C1_07052025_S1_MAMAINES_P6_S6"
    data_home = os.path.normpath("tests/resources/mir_datasets/multivox")
    dataset = multivox.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "C1_07052025_S1_MAMAINES_P6_S6",
        "audio_ortf_l_path": os.path.join(
            data_home,
            "C1/C1_07052025_S1_MAMAINES_P6_S6/audio_ORTF_L_Song1_MAMAINES.wav",
        ),
        "audio_ortf_r_path": os.path.join(
            data_home,
            "C1/C1_07052025_S1_MAMAINES_P6_S6/audio_ORTF_R_Song1_MAMAINES.wav",
        ),
        "video_360_path": os.path.join(
            data_home,
            "C1/C1_07052025_S1_MAMAINES_P6_S6/video360_MAMAINES.mp4",
        ),
        "audio_360_path": os.path.join(
            data_home,
            "C1/C1_07052025_S1_MAMAINES_P6_S6/audio360_Song1_MAMAINES.wav",
        ),
        "near_field_recordings": {
            "S1": os.path.join(
                data_home,
                "C1/C1_07052025_S1_MAMAINES_P6_S6/S1_Song1_MAMAINES.wav",
            ),
            "S2": os.path.join(
                data_home,
                "C1/C1_07052025_S1_MAMAINES_P6_S6/S2_Song1_MAMAINES.wav",
            ),
            "S3": os.path.join(
                data_home,
                "C1/C1_07052025_S1_MAMAINES_P6_S6/S3_Song1_MAMAINES.wav",
            ),
            "S4": os.path.join(
                data_home,
                "C1/C1_07052025_S1_MAMAINES_P6_S6/S4_Song1_MAMAINES.wav",
            ),
            "S5": os.path.join(
                data_home,
                "C1/C1_07052025_S1_MAMAINES_P6_S6/S5_Song1_MAMAINES.wav",
            ),
            "S6": os.path.join(
                data_home,
                "C1/C1_07052025_S1_MAMAINES_P6_S6/S6_Song1_MAMAINES.wav",
            ),
        },
        "song": "MAMAINES",
        "duration": 2.0,
        "key": "A",
        "condition": "C1",
        "recording_space": "Recording Studio",
        "singing_group": "Vocal Chamber Ensemble",
        "singer_positions": {
            "S1": multivox.SingerPosition(
                singer_id="S1",
                facing_direction="C",
                height=172,
                gender="M",
                role="S",
                locations_at_circle=6,
                index_at_circle=0,
            ),
            "S2": multivox.SingerPosition(
                singer_id="S2",
                facing_direction="C",
                height=167,
                gender="M",
                role="S",
                locations_at_circle=6,
                index_at_circle=1,
            ),
            "S3": multivox.SingerPosition(
                singer_id="S3",
                facing_direction="C",
                height=168,
                gender="F",
                role="S",
                locations_at_circle=6,
                index_at_circle=2,
            ),
            "S4": multivox.SingerPosition(
                singer_id="S4",
                facing_direction="C",
                height=168,
                gender="F",
                role="N",
                locations_at_circle=6,
                index_at_circle=3,
            ),
            "S5": multivox.SingerPosition(
                singer_id="S5",
                facing_direction="C",
                height=154,
                gender="F",
                role="S",
                locations_at_circle=6,
                index_at_circle=4,
            ),
            "S6": multivox.SingerPosition(
                singer_id="S6",
                facing_direction="C",
                height=157,
                gender="F",
                role="N",
                locations_at_circle=6,
                index_at_circle=5,
            ),
        },
        "singer_ids": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "singer_count": 6,
        "locations_at_circle": 6,
        "genders": ["M", "M", "F", "F", "F", "F"],
        "facing_directions": ["C", "C", "C", "C", "C", "C"],
        "facing_direction_counts": {"C": 6},
        "unique_facing_directions": ["C"],
        "singers_facing_center": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "singers_facing_outside": [],
        "singers_facing_left": [],
        "singers_facing_right": [],
        "empty_positions": [],
        "singer_heights": ["172", "167", "168", "168", "154", "157"],
        "singer_roles": ["S", "S", "S", "N", "S", "N"],
        "nearfield_ids_captured": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "available_nearfield_ids": ["S1", "S2", "S3", "S4", "S5", "S6"],
    }

    expected_property_types = {
        "audio_ortf_l": tuple,
        "audio_ortf_r": tuple,
        "audio_ortf_stereo": tuple,
        "audio_360": tuple,
        "near_field_audio": dict,
        "angle_increment": float,
        "singer_angular_positions": dict,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # Test audio loading
    audio_l, sr_l = track.audio_ortf_l
    assert sr_l == 22050
    assert audio_l.shape == (44100,)  # mono, 2 seconds at 22050 Hz

    audio_r, sr_r = track.audio_ortf_r
    assert sr_r == 22050
    assert audio_r.shape == (44100,)  # mono, 2 seconds at 22050 Hz

    # Test stereo combination
    audio_stereo, sr_stereo = track.audio_ortf_stereo
    assert sr_stereo == 22050
    assert audio_stereo.shape == (2, 44100)  # 2 channels, 2 seconds at 22050 Hz

    # Test 360 audio
    audio_360, sr_360 = track.audio_360
    assert sr_360 == 22050
    assert audio_360.shape == (4, 44100)  # 4 channels, 2 seconds at 22050 Hz

    # Test near-field recordings
    near_field = track.near_field_audio
    assert len(near_field) == 6  # S1, S2, S3, S4, S5, S6
    assert "S1" in near_field
    audio_s1, sr_s1 = near_field["S1"]
    assert sr_s1 == 22050
    assert audio_s1.shape == (44100,)  # mono, 2 seconds at 22050 Hz
    # Track-level convenience
    assert set(track.available_nearfield_ids) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    audio_s1_b = track.near_field_audio_for("S1")
    assert audio_s1_b is not None

    # Test singer access
    assert track.singer_count == 6
    assert track.singer_ids == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert track.genders == ["M", "M", "F", "F", "F", "F"]
    assert track.facing_directions == [
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
    ]  # From Facing_Direction column
    position_s1 = track.singer_positions["S1"]
    # Position-level convenience (delegates to track)
    assert position_s1.has_near_field(track) is True
    assert position_s1.near_field_path(track) == track.near_field_recordings["S1"]
    audio_s1_c = position_s1.near_field_audio(track)
    assert audio_s1_c is not None
    assert position_s1.singer_id == "S1"
    assert position_s1.height == 172
    assert position_s1.gender == "M"
    assert position_s1.is_female() is False
    assert position_s1.is_male() is True
    assert position_s1.facing_direction == "C"  # From Facing_Direction column
    assert (
        position_s1.locations_at_circle == 6
    )  # From Locations_At_Circle column (count)
    assert position_s1.role == "S"
    assert position_s1.is_at_center() is True  # C = Center (facing toward microphones)
    assert position_s1.is_facing_right() is False
    assert position_s1.is_facing_left() is False
    assert position_s1.is_facing_outside() is False
    assert position_s1.is_empty_position() is False

    # Test angular position
    assert track.angle_increment == 60.0  # 360 / 6 = 60° for 6-singer arrangement
    assert position_s1.angular_position == 0.0  # First position
    assert position_s1.distance_to_far_field == 2.0  # Circle position
    position_s2 = track.singer_positions["S2"]
    assert position_s2.angular_position == 60.0  # Second position
    assert position_s2.distance_to_far_field == 2.0  # Circle position
    angular_positions = track.singer_angular_positions
    assert angular_positions["S1"] == 0.0
    assert angular_positions["S2"] == 60.0
    assert angular_positions["S3"] == 120.0
    assert angular_positions["S4"] == 180.0
    assert angular_positions["S5"] == 240.0
    assert angular_positions["S6"] == 300.0

    # Test facing direction summary properties
    assert track.facing_direction_counts == {"C": 6}
    assert track.unique_facing_directions == ["C"]
    assert track.singers_facing_center == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert track.singers_facing_outside == []
    assert track.singers_facing_left == []
    assert track.singers_facing_right == []
    assert track.empty_positions == []


def test_metadata():
    data_home = os.path.normpath("tests/resources/mir_datasets/multivox")
    dataset = multivox.Dataset(data_home, version="test")
    metadata = dataset._metadata
    assert "C1_07052025_S1_MAMAINES_P6_S6" in metadata
    assert metadata["C1_07052025_S1_MAMAINES_P6_S6"]["Song"] == "MAMAINES"
    # Check that singers are parsed
    track_meta = metadata["C1_07052025_S1_MAMAINES_P6_S6"]
    assert "singers" in track_meta
    assert isinstance(track_meta["singers"], dict)
    assert "S1" in track_meta["singers"]
    assert isinstance(track_meta["singers"]["S1"], multivox.SingerPosition)
    assert track_meta["singers"]["S1"].singer_id == "S1"
    assert "singer_ids" in track_meta
    assert track_meta["singer_ids"] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    # Check that facing_directions is pre-parsed and stored in metadata
    assert "facing_directions" in track_meta
    assert track_meta["facing_directions"] == ["C", "C", "C", "C", "C", "C"]


def test_track_ids():
    data_home = os.path.normpath("tests/resources/mir_datasets/multivox")
    dataset = multivox.Dataset(data_home, version="test")
    track_ids = dataset.track_ids
    assert "C1_07052025_S1_MAMAINES_P6_S6" in track_ids
    assert "C1_01042025_S1_VAMOSAREMAR_P17_S16" in track_ids
    assert "C1_09042025_S1_VAMOSAREMAR_P6_S6" in track_ids
    assert "C1_11042025_S1_PEPECHIQUITO_P7_S6" in track_ids


def test_track_with_empty_position():
    """Test a track with an empty position (E in Facing_Direction)."""
    track_id = "C3_14052025_S1_AGUACERO_FRAGMENT_P6_S5"
    data_home = os.path.normpath("tests/resources/mir_datasets/multivox")
    dataset = multivox.Dataset(data_home, version="test")
    track = dataset.track(track_id)

    # Verify basic track properties
    assert track.track_id == track_id
    assert track.song == "AGUACERO FRAGMENT"
    assert track.locations_at_circle == 6
    assert track.singer_count == 5  # S1, S2, S3, S4, S6 (S5 is empty)

    # Verify singer IDs (S5 should still be in the list but marked as empty)
    assert "S1" in track.singer_ids
    assert "S2" in track.singer_ids
    assert "S3" in track.singer_ids
    assert "S4" in track.singer_ids
    assert "S5" in track.singer_ids  # Empty position, but ID is present
    assert "S6" in track.singer_ids

    # Verify facing directions include None for empty position (was "E" in CSV)
    assert None in track.facing_directions
    assert track.facing_directions == ["C", "C", "C", "C", None, "C"]

    # Verify empty_positions property
    assert track.empty_positions == ["S5"]
    assert "S5" in track.empty_positions

    # Verify the empty position object
    position_s5 = track.singer_positions["S5"]
    assert position_s5.facing_direction is None
    assert position_s5.is_empty_position() is True
    assert position_s5.is_at_center() is False
    assert position_s5.is_facing_outside() is False
    assert position_s5.is_facing_left() is False
    assert position_s5.is_facing_right() is False

    # In this track, position S5 is empty (no singer assigned to that position
    # in the circular arrangement), so we expect None: 1 in counts.
    # Verify facing direction counts: Empty positions (unoccupied positions
    # in the circle) have None as their facing direction (originally marked as
    # "E" in the CSV). The facing_direction_counts property uses Counter()
    # on the facing_directions list, which includes None values. Therefore,
    # None will appear as a key in the counts dictionary when empty positions
    # are present.
    assert None in track.facing_direction_counts
    assert track.facing_direction_counts[None] == 1
    assert track.facing_direction_counts["C"] == 5

    # Verify unique facing directions
    assert None in track.unique_facing_directions
    assert "C" in track.unique_facing_directions

    # Verify that empty positions are not in other facing direction lists
    assert "S5" not in track.singers_facing_center
    assert "S5" not in track.singers_facing_outside
    assert "S5" not in track.singers_facing_left
    assert "S5" not in track.singers_facing_right

    # Verify angular position for empty position (should still be computed)
    # Position index 4 (S5) should be at 240° (4 * 60°)
    assert track.angle_increment == 60.0
    assert position_s5.angular_position == 240.0
    assert position_s5.distance_to_far_field == 2.0  # Still part of circle
    angular_positions = track.singer_angular_positions
    assert angular_positions["S5"] == 240.0


def test_track_16_singers():
    track_id = "C1_01042025_S1_VAMOSAREMAR_P17_S16"
    data_home = os.path.normpath("tests/resources/mir_datasets/multivox")
    dataset = multivox.Dataset(data_home, version="test")
    track = dataset.track(track_id)

    assert track.singer_count == 16
    assert track.locations_at_circle == 16
    assert len(track.singer_ids) == 16
    assert track.singer_ids[0] == "S1"
    assert track.singer_ids[15] == "S16"
    assert track.angle_increment == 22.5
    assert track.singer_positions["S1"].angular_position == 0.0
    assert track.singer_positions["S2"].angular_position == 22.5
    # S3 has role "N" (Non-singing)
    assert track.singer_positions["S3"].role == "N"
    assert track.singer_positions["S1"].role == "S"


def test_track_missing_nearfield():
    # Track has 6 positions, but S3 is missing from Nearfield_Files_Captured
    track_id = "C1_09042025_S1_VAMOSAREMAR_P6_S6"
    data_home = os.path.normpath("tests/resources/mir_datasets/multivox")
    dataset = multivox.Dataset(data_home, version="test")
    track = dataset.track(track_id)

    assert track.singer_count == 6  # Should still have 6 singers
    assert track.locations_at_circle == 6
    assert track.singer_ids == ["S1", "S2", "S3", "S4", "S5", "S6"]

    # S3 should have metadata but NO near-field recording
    assert "S3" in track.singer_positions
    assert track.singer_positions["S3"].height == 168
    assert "S3" not in track.near_field_recordings

    # S4 should have correct metadata (aligned by position, not by nearfield list index)
    # Metadata for S4 is role "S", height 154
    assert track.singer_positions["S4"].role == "S"
    assert track.singer_positions["S4"].height == 154


def test_track_with_instructor():
    # Track has 6 circle positions + Sinstructor
    track_id = "C1_11042025_S1_PEPECHIQUITO_P7_S6"
    data_home = os.path.normpath("tests/resources/mir_datasets/multivox")
    dataset = multivox.Dataset(data_home, version="test")
    track = dataset.track(track_id)

    assert track.singer_count == 7
    assert track.locations_at_circle == 6
    assert "Sinstructor" in track.singer_ids
    assert "Sinstructor" in track.singer_positions

    # Sinstructor is NOT at circle
    assert track.singer_positions["Sinstructor"].locations_at_circle is None
    assert track.singer_positions["Sinstructor"].index_at_circle is None
    assert track.singer_positions["Sinstructor"].angular_position is None
    assert (
        track.singer_positions["Sinstructor"].distance_to_far_field is None
    )  # Non-circle

    # Circle singers still have correct circle data
    assert track.singer_positions["S1"].locations_at_circle == 6
    assert track.singer_positions["S1"].angular_position == 0.0

    # Near-field convenience:
    # In this test dataset, Sinstructor IS in near_field_recordings
    assert "Sinstructor" in track.near_field_recordings
    assert track.singer_positions["Sinstructor"].near_field_path(track) is not None
    assert track.singer_positions["Sinstructor"].has_near_field(track) is True
    # Note: audio won't load because test index uses placeholder paths
    # Circle singer also has near-field path set
    assert track.singer_positions["S1"].near_field_path(track) is not None
    assert track.singer_positions["S1"].has_near_field(track) is True


def test_singer_position_repr_and_eq():
    pos = multivox.SingerPosition(
        singer_id="S1",
        facing_direction=None,
        height=None,
        gender=None,
        role=None,
        locations_at_circle=None,
        index_at_circle=None,
    )
    assert pos != "not-a-position"
    assert pos.__eq__("not-a-position") is False
    repr_str = repr(pos)
    assert "SingerPosition" in repr_str
    assert "S1" in repr_str


def test_angle_increment_none(tmp_path):
    track = _make_track(tmp_path, {"locations_at_circle": None, "singer_ids": []}, {})
    assert track.angle_increment is None


def test_nearfield_keys(tmp_path):
    paths = {
        "near_field_S1": ("file1.wav", "x"),
        "near_field_S2": ("file2.wav", "y"),
    }
    meta = {"singer_ids": ["S1", "S2"], "locations_at_circle": 2}
    track = _make_track(tmp_path, meta, paths)
    assert track.near_field_recordings["S1"].endswith("file1.wav")
    assert track.near_field_recordings["S2"].endswith("file2.wav")
    assert set[str](track.available_nearfield_ids) == {"S1", "S2"}
    # Files don't exist, so accessing near_field_audio should raise an exception
    with pytest.raises(FileNotFoundError):
        _ = track.near_field_audio


def test_nearfield_key_with_none_path(tmp_path):
    paths = {
        "near_field_S1": (None, "checksum"),
    }
    meta = {"singer_ids": ["S1"], "locations_at_circle": 1}
    track = _make_track(tmp_path, meta, paths)
    assert "S1" not in track.near_field_recordings


def test_empty_fields_properties(tmp_path):
    meta = {
        "Singer_Height": "",
        "Singer_Roles": "",
        "Nearfield_Files_Captured": "",
        "singer_ids": [],
        "locations_at_circle": 0,
    }
    track = _make_track(tmp_path, meta, {})
    assert track.singer_heights == []
    assert track.singer_roles == []
    assert track.nearfield_ids_captured == []

    meta2 = {
        "singer_ids": [],
        "locations_at_circle": 0,
    }
    track2 = _make_track(tmp_path, meta2, {})
    assert track2.singer_heights == []
    assert track2.singer_roles == []
    assert track2.nearfield_ids_captured == []


def test_audio_none_branches(tmp_path):
    meta = {"singer_ids": [], "locations_at_circle": 0}

    # Required audio should raise when paths are missing
    track = _make_track(tmp_path, meta, {"audio_ortf_l": (None, None)})
    with pytest.raises(FileNotFoundError):
        _ = track.audio_ortf_l

    track2 = _make_track(tmp_path, meta, {"audio_ortf_r": (None, None)})
    with pytest.raises(FileNotFoundError):
        _ = track2.audio_ortf_r

    # audio_360 should return None when path is None (optional property)
    track3 = _make_track(tmp_path, meta, {})
    assert track3.audio_360 is None

    track4 = _make_track(tmp_path, meta, {"audio_ortf_l": (None, None)})
    with pytest.raises(FileNotFoundError):
        _ = track4.audio_ortf_stereo

    track5 = _make_track(tmp_path, meta, {})
    with pytest.raises(FileNotFoundError):
        _ = track5.audio_ortf_stereo
    assert track5.near_field_audio == {}


def test_audio360_load_branch(tmp_path):
    audio_path = tmp_path / "a.wav"
    with wave.open(str(audio_path), "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8)

    paths = {"audio_360": (os.fspath(audio_path), "checksum")}
    meta = {"singer_ids": [], "locations_at_circle": 0}
    track = _make_track(tmp_path, meta, paths)
    audio = track.audio_360
    assert audio is not None
    data, sr = audio
    assert sr == 8000
    assert data.shape[0] == 2


def test_metadata_missing_file(tmp_path):
    ds = multivox.Dataset(data_home=str(tmp_path), version="sample")
    with pytest.raises(FileNotFoundError):
        _ = ds._metadata


def test_metadata_invalid_height_and_locations(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "Date,Session_ID,Recording_Space,Song,Path,Duration,Key,Singing_Group,Vocal_Content,Condition,Locations_At_Circle,Empty_Circle_Locations,Instructor_Visible,Visible_People,Degree,Facing_Direction,Singer_Roles,Singer_Height,Gender,Nearfield_Files_Captured\n"
        "01/01/2025,1A,Auditorium,TEST,TID,1.0,C,Choir,Mixed,C1,abc,0,N,0,NA, , ,abc,M, \n"
    )
    ds = multivox.Dataset(data_home=str(tmp_path), version="sample")
    metadata = ds._metadata
    entry = metadata["TID"]
    assert entry["locations_at_circle"] is None
    assert entry["singer_ids"] == []


def test_metadata_valueerror_in_parsing(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "Date,Session_ID,Recording_Space,Song,Path,Duration,Key,Singing_Group,Vocal_Content,Condition,Locations_At_Circle,Empty_Circle_Locations,Instructor_Visible,Visible_People,Degree,Facing_Direction,Singer_Roles,Singer_Height,Gender,Nearfield_Files_Captured\n"
        "01/01/2025,1A,Auditorium,TEST,TID,1.0,C,Choir,Mixed,C1,6,0,N,6,NA,C,C,C,invalid_height,M,S1\n"
    )
    ds = multivox.Dataset(data_home=str(tmp_path), version="sample")
    metadata = ds._metadata
    entry = metadata["TID"]
    assert entry["singers"]["S1"].height is None


def test_metadata_valueerror_locations_at_circle(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "Date,Session_ID,Recording_Space,Song,Path,Duration,Key,Singing_Group,Vocal_Content,Condition,Locations_At_Circle,Empty_Circle_Locations,Instructor_Visible,Visible_People,Degree,Facing_Direction,Singer_Roles,Singer_Height,Gender,Nearfield_Files_Captured\n"
        "01/01/2025,1A,Auditorium,TEST,TID,1.0,C,Choir,Mixed,C1,6,0,N,6,NA,C,C,C,172,M,S1\n"
    )
    original_int = int
    call_count = [0]

    def mock_int(x):
        call_count[0] += 1
        if call_count[0] == 1 and x == "6":
            raise ValueError("test")
        return original_int(x)

    with patch("builtins.int", side_effect=mock_int):
        ds = multivox.Dataset(data_home=str(tmp_path), version="sample")
        metadata = ds._metadata
        entry = metadata["TID"]
        assert entry["locations_at_circle"] is None
