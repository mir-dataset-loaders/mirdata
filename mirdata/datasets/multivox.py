"""MULTIVOX Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    MULTIVOX is a multimodal, spatial audio–visual dataset of a-cappella vocal performances
    recorded in controlled conditions with both choir and vocal chamber ensembles. The dataset
    comprises 154 performances (≈3 hours total), captured in two acoustically distinct spaces
    (auditorium and recording studio).

    Each performance includes synchronized 360° video, far-field audio (ORTF stereo with separate
    left and right channel files, plus 360° camera audio from the Insta360 X3), and per-singer
    near-field recordings captured on personal devices.
    Performances feature 6– and 16-singer configurations arranged in a circle around the 360°
    camera and far-field devices. The repertoire covers 18 short choral pieces, including vocal
    warm-ups, Latin American songs, and arranged popular music.

    The dataset is designed to support research on spatial audio, sound event localization,
    source separation, ensemble synchronization, and multimodal modeling of group singing.

    It was created by G. Meza, M. Sepúlveda, A. S. Roman, J. R. Sigal Sefchovich, and I. R. Roman
    in 2025.

    For more information, visit: https://zenodo.org/records/17058101

    The dataset is licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0).

.. admonition:: Usage Examples
    :class: dropdown

    **Loading a track and accessing audio**

    .. code-block:: python

        import mirdata

        dataset = mirdata.initialize('multivox')
        track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')

        # Far-field mixture (ORTF stereo)
        mixture, sr = track.audio_ortf_stereo

        # Angular positions for each singer
        for sid, angle in track.singer_angular_positions.items():
            print(f"{sid}: {angle:.1f}°")

    **Loading near-field recordings**

    .. code-block:: python

        for sid in track.singer_ids:
            pos = track.singer_positions[sid]
            if pos.has_near_field(track):
                audio, sr = pos.near_field_audio(track)
                print(f"{sid}: {audio.shape[0]/sr:.1f}s")

    **Accessing position metadata**

    .. code-block:: python

        for sid, pos in track.singer_positions.items():
            print(f"{sid}: {pos.angular_position}°, facing={pos.facing_direction}")

        # Filter by gender
        females = [sid for sid in track.singer_ids
                   if track.singer_positions[sid].is_female()]

"""

import csv
import os
from collections import Counter
from typing import BinaryIO, Dict, List, Optional, Tuple

import librosa
import cv2
import numpy as np
from smart_open import open

from mirdata import core, download_utils, io


BIBTEX = """
@inproceedings{meza2025multivox,
 author = {Meza, G. and Sepúlveda, M. and Roman, A. S. and Sigal Sefchovich, J. R. and Roman, I. R.},
 title = {{MULTIVOX: A Spatial Audio-Visual Dataset of Singing Groups}},
 booktitle = {Proceedings of the AES International Conference on Artificial Intelligence and Machine Learning for Audio (AES AIMLA)},
 year = {2025},
 address = {London},
 doi = {10.5281/zenodo.17058101}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="mirdata_multivox_index_1.0.json",
        url="https://zenodo.org/records/18274999/files/mirdata_multivox_index_1.0.json?download=1",
        checksum="88d49145da4f30d4fc60ef4d31b611ac",
    ),
    "sample": core.Index(filename="mirdata_multivox_index_sample.json"),
}

REMOTES = {
    "C1": download_utils.RemoteFileMetadata(
        filename="C1.zip",
        url="https://zenodo.org/records/17058101/files/C1.zip?download=1",
        checksum="2332b9d1e3c2bbb176e539417a80c21a",
    ),
    "C2": download_utils.RemoteFileMetadata(
        filename="C2.zip",
        url="https://zenodo.org/records/17058101/files/C2.zip?download=1",
        checksum="fed5ede02695b5a91453276b454fe148",
    ),
    "C3": download_utils.RemoteFileMetadata(
        filename="C3.zip",
        url="https://zenodo.org/records/17065497/files/C3.zip?download=1",
        checksum="cf1edc9032e4050b183666566c3c1913",
    ),
    "C4": download_utils.RemoteFileMetadata(
        filename="C4.zip",
        url="https://zenodo.org/records/17065497/files/C4.zip?download=1",
        checksum="5510d6624e8823400dace806c31776fb",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="metadata.csv",
        url="https://zenodo.org/records/17058101/files/metadata.csv?download=1",
        checksum="81c154d7c6f5974c1ff7076faf32c24d",
    ),
    "readme": download_utils.RemoteFileMetadata(
        filename="README.md",
        url="https://zenodo.org/records/17058101/files/README.md?download=1",
        checksum="4d3f1de5801965183b0beae99222d4a9",
    ),
    "description_pdf": download_utils.RemoteFileMetadata(
        filename="MULTIVOX_extended_dataset_description_and_supplement.pdf",
        url="https://zenodo.org/records/17058101/files/MULTIVOX_extended_dataset_description_and_supplement.pdf?download=1",
        checksum="db218a963ad9546c628660cd92daa488",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International (CC-BY-4.0)"


class SingerPosition:
    """MULTIVOX SingerPosition class - encapsulates metadata for a physical position.

    A SingerPosition represents a physical location in the circular arrangement around
    the recording equipment. It may be occupied by a singer or empty (no performer
    assigned). For non-circle participants like "Sinstructor", spatial attributes are None.

    Args:
        singer_id (str): Position identifier (e.g., "S1", "S2", "Sinstructor")
        facing_direction (str, optional): Facing direction code:

            - ``'C'`` = Center: facing toward the microphones/center
            - ``'O'`` = Outside: facing away from the microphones (outward)
            - ``'L'`` = Left: facing counterclockwise
            - ``'R'`` = Right: facing clockwise
            - ``None`` = Empty position or non-circle participant

        height (int, optional): Singer height in cm (None if position is empty)
        gender (str, optional): Gender code: ``'M'`` (male), ``'F'`` (female), or None
        role (str, optional): Role code: ``'S'`` (singing), ``'N'`` (non-singing/silent)
        locations_at_circle (int, optional): Total positions in the circle geometry
            (6 or 16). None for non-circle participants (e.g., Sinstructor).
        index_at_circle (int, optional): Zero-based index around the circle (0 to N-1).
            Used to compute ``angular_position``. None for non-circle participants.

    Attributes:
        singer_id (str): Position identifier (e.g., "S1", "S2", "Sinstructor")
        facing_direction (str or None): Facing direction code (C/O/L/R) or None if empty
        height (int or None): Singer height in cm
        gender (str or None): Gender code (M/F)
        role (str or None): Role code (S=singing, N=silent)
        locations_at_circle (int or None): Circle geometry size (6 or 16)
        index_at_circle (int or None): Zero-based position index around the circle
        angular_position (float or None): Computed angular position in degrees (0–360)
        distance_to_far_field (float or None): Distance to far-field microphones in meters.
            2.0m for circle positions, None for non-circle participants.

    Examples:

        **Basic usage - accessing position metadata:**

        >>> import mirdata
        >>> dataset = mirdata.initialize('multivox')
        >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
        >>> position = track.singer_positions['S1']
        >>> position.gender          # 'M'
        >>> position.height           # 172
        >>> position.angular_position # 0.0 (first position, in front of mics)

        **Loading near-field audio for a position:**

        >>> if position.has_near_field(track):
        ...     audio, sr = position.near_field_audio(track)
        ...     print(f"Loaded {audio.shape[0]/sr:.1f}s at {sr}Hz")

        **Iterating over all positions:**

        >>> for sid in track.singer_ids:
        ...     pos = track.singer_positions[sid]
        ...     if pos.is_empty_position():
        ...         print(f"{sid}: empty position")
        ...     else:
        ...         print(f"{sid}: {pos.gender}, {pos.angular_position:.0f}°")

        **Filtering by attributes:**

        >>> # Get all female singers facing center
        >>> center_females = [
        ...     track.singer_positions[sid]
        ...     for sid in track.singer_ids
        ...     if track.singer_positions[sid].is_female()
        ...     and track.singer_positions[sid].is_at_center()
        ... ]
    """

    def __init__(
        self,
        singer_id: str,
        facing_direction: Optional[str] = None,
        height: Optional[int] = None,
        gender: Optional[str] = None,
        role: Optional[str] = None,
        locations_at_circle: Optional[int] = None,
        index_at_circle: Optional[int] = None,
    ):
        self.singer_id = singer_id
        self.facing_direction = facing_direction
        self.height = height
        self.gender = gender
        self.role = role
        self.locations_at_circle = locations_at_circle
        self.index_at_circle = index_at_circle  # Zero-based index around the circle

    def is_female(self) -> bool:
        """Check if the occupant is female.

        Returns:
            bool: True if gender is 'F', False otherwise
        """
        return self.gender == "F"

    def is_male(self) -> bool:
        """Check if the occupant is male.

        Returns:
            bool: True if gender is 'M', False otherwise
        """
        return self.gender == "M"

    def is_at_center(self) -> bool:
        """Check if the singer at this position is facing toward the center/microphones.

        Returns:
            bool: True if facing_direction is 'C' (Center), False otherwise
        """
        return self.facing_direction == "C"

    def is_facing_outside(self) -> bool:
        """Check if the singer at this position is facing away from the microphones (outward).

        Returns:
            bool: True if facing_direction is 'O' (Outside), False otherwise
        """
        return self.facing_direction == "O"

    def is_facing_left(self) -> bool:
        """Check if the singer at this position is facing counterclockwise (to their left).

        Returns:
            bool: True if facing_direction is 'L' (Left), False otherwise
        """
        return self.facing_direction == "L"

    def is_facing_right(self) -> bool:
        """Check if the singer at this position is facing clockwise (to their right).

        Returns:
            bool: True if facing_direction is 'R' (Right), False otherwise
        """
        return self.facing_direction == "R"

    def is_empty_position(self) -> bool:
        """Check if this position is empty (no singer assigned).

        Returns:
            bool: True if the position is empty, False otherwise
        """
        # A position is empty if it's part of the circle (locations_at_circle is set)
        # but has no facing direction (which is set to None for 'E' or missing values)
        return self.facing_direction is None and self.locations_at_circle is not None

    @property
    def angular_position(self) -> Optional[float]:
        """Get the angular position of this singer position in degrees.

        Angular positions are evenly spaced around the circle, starting at 0°
        (in front of the far-field microphones and 360° camera). The increment
        is 360° / locations_at_circle (22.5° for 16 singers, 60° for 6 singers).

        Returns:
            float: Angular position in degrees (0-360), where 0° is in front
                of the far-field microphones. Returns None if index_at_circle or
                locations_at_circle is not available.

        Example:
            >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
            >>> position = track.singer_positions['S1']
            >>> print(position.angular_position)  # 0.0 (first position)
            >>> position2 = track.singer_positions['S2']
            >>> print(position2.angular_position)  # 60.0 (for 6-singer arrangement)
        """
        if self.index_at_circle is None or self.locations_at_circle is None:
            return None
        angle_increment = 360.0 / self.locations_at_circle
        return self.index_at_circle * angle_increment

    @property
    def distance_to_far_field(self) -> Optional[float]:
        """Get the distance from this position to the far-field microphones in meters.

        The far-field recording equipment (ORTF stereo and 360° camera) was
        strictly positioned at 2.0 meters from the circle of singers.

        Returns:
            float: Distance in meters (2.0) for circle positions, or None for
                non-circle participants (e.g., Sinstructor, who is positioned
                outside the circle at an unspecified distance > 2 meters).
        """
        if self.locations_at_circle is None:
            return None  # Non-circle participant (e.g., instructor)
        return 2.0

    def __eq__(self, other):
        """Check equality of SingerPosition objects."""
        if not isinstance(other, SingerPosition):
            return False
        return (
            self.singer_id == other.singer_id
            and self.facing_direction == other.facing_direction
            and self.height == other.height
            and self.gender == other.gender
            and self.role == other.role
            and self.locations_at_circle == other.locations_at_circle
            and self.index_at_circle == other.index_at_circle
        )

    def __repr__(self) -> str:
        return (
            f"SingerPosition(singer_id='{self.singer_id}', "
            f"gender={self.gender}, height={self.height}, "
            f"facing_direction={self.facing_direction})"
        )

    # --- Near-field convenience (requires Track) ---
    def near_field_path(self, track: "Track") -> Optional[str]:
        """Get the near-field file path for this position (if present in the index).

        Args:
            track (Track): The parent Track instance (provides index-resolved paths).

        Returns:
            str or None: Absolute path to the near-field file if available, else None.
        """
        return track.near_field_recordings.get(self.singer_id)

    def has_near_field(self, track: "Track") -> bool:
        """Check if this position has a near-field recording in the index."""
        return self.near_field_path(track) is not None

    def near_field_audio(self, track: "Track") -> Optional[Tuple[np.ndarray, float]]:
        """Load near-field audio for this position via the track cache.

        Args:
            track (Track): The parent Track instance (provides cached audio).

        Returns:
            tuple or None: (audio, sample_rate) if available; None otherwise.
        """
        # track.near_field_audio is cached; reuse it instead of reloading
        return track.near_field_audio.get(self.singer_id)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a MULTIVOX audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file

    Returns:
        * np.ndarray - audio signal
        * float - sample rate
    """
    return librosa.load(fhandle, sr=None, mono=False)


def load_video(video_path, target_fps=None, frame_size=None):
    """Load a MULTIVOX video file.

    Args:
        video_path (str): Path to video file
        target_fps (float, optional): Target frames per second for resampling.
            If None, uses original fps.
        frame_size (tuple, optional): Target frame size as (height, width).
            If None, uses original frame size.

    Returns:
        * np.ndarray - video array with shape (T, H, W, 3) where T is number of frames

    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # If frame_size is given, resize the frame
        if frame_size is not None:
            target_h, target_w = frame_size  # (H, W)
            frame_bgr = cv2.resize(
                frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    data = np.array(frames)

    if target_fps is not None:
        # Get original fps
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps is None or orig_fps <= 0:
            orig_fps = 30.0  # fallback
        T = data.shape[0]
        duration = T / orig_fps
        T_new = int(np.floor(duration * target_fps))
        idx = np.round((np.arange(T_new) / target_fps) * orig_fps).astype(int)
        idx = np.clip(idx, 0, T - 1)
        video_array = data[idx]
    else:
        video_array = data

    cap.release()
    return video_array  # shape: (T, H, W, 3)


class Track(core.Track):
    """MULTIVOX Track class - represents a single vocal performance recording.

    A Track encapsulates all data for one performance: far-field mixtures (ORTF stereo,
    360° camera audio), per-singer near-field recordings, 360° video, and rich metadata
    about the spatial arrangement and singer characteristics.

    Args:
        track_id (str): Unique track identifier (e.g., ``'C1_07052025_S1_MAMAINES_P6_S6'``)

    Attributes:
        track_id (str): Unique track identifier
        audio_360_path (str or None): Path to 360° camera audio (stereo, 48kHz from Insta360 X3).
            None for Session 1A tracks (01/04/2025) where audio_360 was not captured.
        audio_ortf_l_path (str or None): Path to ORTF left channel audio file
        audio_ortf_r_path (str or None): Path to ORTF right channel audio file
        video_360_path (str or None): Path to 360° video file
        near_field_recordings (dict): Mapping of IDs to near-field audio file paths

    Properties (Metadata):
        condition (str): Experimental condition code (C1, C2, C3, or C4)
        recording_space (str): ``'Auditorium'`` or ``'Recording Studio'``
        singing_group (str): ``'Choir'`` or ``'Vocal Chamber Ensemble'``
        song (str): Song/piece name
        key (str): Tonal center of the performance
        duration (float): Duration in seconds

    Properties (Spatial Arrangement):
        locations_at_circle (int): Circle geometry size (6 or 16 positions)
        angle_increment (float): Angular spacing between positions in degrees
            (60° for 6-singer, 22.5° for 16-singer arrangements)
        singer_ids (list[str]): All position IDs in order (e.g., ``['S1', 'S2', ...]``)
        singer_positions (dict[str, SingerPosition]): Mapping of IDs to :class:`SingerPosition` objects
        singer_angular_positions (dict[str, float]): Mapping of IDs to angular positions (0–360°)
        singer_count (int): Number of actual singers (excluding empty positions)

    Properties (Facing Directions):
        facing_directions (list[str or None]): Facing codes in position order
        unique_facing_directions (list[str or None]): Unique facing codes present
        facing_direction_counts (dict[str, int]): Count per direction
        singers_facing_center (list[str]): IDs facing center (``'C'``)
        singers_facing_outside (list[str]): IDs facing outward (``'O'``)
        singers_facing_left (list[str]): IDs facing counterclockwise (``'L'``)
        singers_facing_right (list[str]): IDs facing clockwise (``'R'``)
        empty_positions (list[str]): IDs at unoccupied positions

    Properties (Demographics):
        genders (list[str]): Gender codes in position order (``'M'`` or ``'F'``)

    Properties (CSV Metadata):
        singer_heights (list[str]): Height entries from CSV (split/stripped)
        singer_roles (list[str]): Role entries from CSV (split/stripped)
        nearfield_ids_captured (list[str]): Nearfield ID list from CSV (split/stripped)

    Properties (Near-field Access):
        available_nearfield_ids (list[str]): IDs with near-field recordings in the index

    Properties (Audio - loaded lazily on first access):
        audio_ortf_l (tuple): ORTF left channel ``(np.ndarray, sample_rate)``
        audio_ortf_r (tuple): ORTF right channel ``(np.ndarray, sample_rate)``
        audio_ortf_stereo (tuple): Combined ORTF stereo ``(np.ndarray[2, N], sample_rate)``
        audio_360 (tuple or None): 360° camera audio ``(np.ndarray[2, N], sample_rate)``
            Returns None if not available (e.g., Session 1A tracks).
            Check ``if track.audio_360 is not None`` before using.

    Cached Properties:
        near_field_audio (dict): Mapping of IDs to ``(np.ndarray, sample_rate)`` tuples

    Examples:

        **Quick start - loading a track:**

        >>> import mirdata
        >>> dataset = mirdata.initialize('multivox')
        >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
        >>> print(track.song, track.singing_group)
        'MAMAINES' 'Vocal Chamber Ensemble'

        **Accessing far-field audio (mixture):**

        >>> # ORTF stereo mixture (L+R channels)
        >>> mixture, sr = track.audio_ortf_stereo
        >>> print(f"Shape: {mixture.shape}, Sample rate: {sr}Hz")
        Shape: (2, 2646000), Sample rate: 22050Hz

        >>> # 360° camera audio (if available)
        >>> audio_360_data = track.audio_360
        >>> if audio_360_data is not None:
        ...     audio_360, sr_360 = audio_360_data
        ...     print(f"360° audio: {audio_360.shape}")

        **Accessing per-singer near-field recordings:**

        >>> # Method 1: Via SingerPosition (recommended)
        >>> for sid in track.singer_ids:
        ...     pos = track.singer_positions[sid]
        ...     if pos.has_near_field(track):
        ...         audio, sr = pos.near_field_audio(track)
        ...         print(f"{sid}: {audio.shape[0]/sr:.1f}s, angle={pos.angular_position}°")

        >>> # Method 2: Via Track directly
        >>> for sid in track.available_nearfield_ids:
        ...     audio, sr = track.near_field_audio_for(sid)
        ...     print(f"{sid}: loaded {audio.shape}")

        **Working with spatial metadata:**

        >>> # Get angular positions for source localization
        >>> print(track.singer_angular_positions)
        {'S1': 0.0, 'S2': 60.0, 'S3': 120.0, 'S4': 180.0, 'S5': 240.0, 'S6': 300.0}

        >>> # Filter by facing direction
        >>> center_facing = track.singers_facing_center
        >>> print(f"{len(center_facing)} singers facing center")

        >>> # Analyze facing distribution
        >>> print(track.facing_direction_counts)
        {'C': 6}

        **Handling empty positions and special participants:**

        >>> # Tracks may have empty positions (no singer assigned)
        >>> if track.empty_positions:
        ...     print(f"Empty positions: {track.empty_positions}")

        >>> # Some tracks have non-circle participants (e.g., instructor)
        >>> if 'Sinstructor' in track.singer_positions:
        ...     instructor = track.singer_positions['Sinstructor']
        ...     print(f"Instructor has near-field: {instructor.has_near_field(track)}")

    See Also:
        :class:`SingerPosition`: Per-position metadata and near-field access helpers.
        :class:`Dataset`: Dataset-level operations (download, validate, iteration).
    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_ortf_l_path = (
            self.get_path("audio_ortf_l")
            if "audio_ortf_l" in self._track_paths
            else None
        )
        self.audio_ortf_r_path = (
            self.get_path("audio_ortf_r")
            if "audio_ortf_r" in self._track_paths
            else None
        )
        self.audio_360_path = (
            self.get_path("audio_360") if "audio_360" in self._track_paths else None
        )
        self.video_360_path = (
            self.get_path("video_360") if "video_360" in self._track_paths else None
        )

        # Get near-field recordings from index
        # Uses keys (near_field_S1, near_field_Sinstructor, ...)
        self.near_field_recordings = {}
        for key, value in self._track_paths.items():
            if key.startswith("near_field_"):
                singer_id = key.replace("near_field_", "")
                if isinstance(value, (list, tuple)) and len(value) >= 1:
                    path = value[0]
                    if path is not None:
                        self.near_field_recordings[singer_id] = os.path.join(
                            self._data_home, path
                        )

    @property
    def angle_increment(self) -> Optional[float]:
        """Get the angular increment between positions in degrees.

        Positions are evenly spaced around the circle. For 16-singer arrangements,
        the increment is 22.5° (360° / 16). For 6-singer arrangements, it's 60°
        (360° / 6).

        Returns:
            float: Angular increment in degrees, or None if locations_at_circle
                is not available.

        Example:
            >>> track = dataset.track('C1_01042025_S1_VAMOSAREMAR_P17_S16')
            >>> print(track.angle_increment)  # 22.5 (for 16-singer arrangement)
        """
        if self.locations_at_circle is None:
            return None
        return 360.0 / self.locations_at_circle

    @property
    def condition(self) -> Optional[str]:
        return self._track_metadata.get("Condition")

    @property
    def duration(self) -> Optional[float]:
        dur = self._track_metadata.get("Duration")
        return float(dur) if dur else None

    @property
    def empty_positions(self) -> List[str]:
        """Get list of IDs at empty positions (no singer assigned).

        Returns:
            list: List of IDs at empty positions (facing_direction is None).
        """
        return [
            sid
            for sid in self.singer_ids
            if sid in self.singer_positions
            and self.singer_positions[sid].is_empty_position()
        ]

    @property
    def facing_direction_counts(self) -> Dict[str, int]:
        """Get count of singers facing each direction.

        Returns:
            dict: Dictionary mapping facing direction codes to counts.
                Keys are: 'C' (Center), 'O' (Outside), 'L' (Left), 'R' (Right), 'E' (Empty).

        Example:
            >>> track = dataset.track('C1_01042025_S1_VAMOSAREMAR_P17_S16')
            >>> print(track.facing_direction_counts)  # {'C': 16}
            >>> # For C4 tracks with varied arrangements:
            >>> print(track.facing_direction_counts)  # {'C': 8, 'R': 3, 'O': 3, 'L': 2}
        """
        return dict[str, int](Counter[str](self.facing_directions))

    @property
    def facing_directions(self) -> List[str]:
        """Get list of facing directions in singer order.

        Returns:
            list: List of facing direction code strings in the same order as singer_ids.
                Codes are:
                - C=Center: facing toward microphones/center
                - O=Outside: facing away from microphones (outward)
                - L=Left: facing counterclockwise (to their left)
                - R=Right: facing clockwise (to their right)
                - E=Empty: no singer assigned to this position

        Example:
            >>> track = dataset.track('C1_01042025_S1_VAMOSAREMAR_P17_S16')
            >>> print(track.facing_directions)  # ['C', 'C', 'C', 'C', ...]
        """
        # Use pre-parsed list from metadata (parsed from CSV during Dataset._metadata)
        return self._track_metadata.get("facing_directions", [])

    @property
    def genders(self) -> List[str]:
        """Get list of genders in position order.

        Returns:
            list: List of gender strings ('M' or 'F') in the same order as singer_ids.

        Example:
            >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
            >>> print(track.genders)  # ['M', 'M', 'F', 'F', 'F', 'F']
            >>> num_females = track.genders.count('F')
        """
        results: List[str] = []
        for sid in self.singer_ids:
            if sid in self.singer_positions:
                gender = self.singer_positions[sid].gender
                if gender is not None:
                    results.append(gender)
        return results

    @property
    def key(self) -> Optional[str]:
        return self._track_metadata.get("Key")

    @property
    def locations_at_circle(self) -> Optional[int]:
        """Get the number of positions in the circle layout (6 or 16)."""
        return self._track_metadata.get("locations_at_circle")

    @property
    def nearfield_ids_captured(self) -> List[str]:
        """Get nearfield singer IDs from CSV (split/stripped).

        Returns:
            list: List of singer IDs from Nearfield_Files_Captured column.
        """
        raw = self._track_metadata.get("Nearfield_Files_Captured")
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]

    @property
    def recording_space(self) -> Optional[str]:
        return self._track_metadata.get("Recording_Space")

    @property
    def singer_angular_positions(self) -> Dict[str, float]:
        """Get angular positions for all positions in degrees.

        Angular positions are evenly spaced around the circle, starting at 0°
        (in front of the far-field microphones and 360° camera). Each position's
        location corresponds to its index in the singer_ids list.

        Returns:
            dict: Dictionary mapping IDs to angular positions in degrees (0-360).

        Example:
            >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
            >>> positions = track.singer_angular_positions
            >>> print(positions['S1'])  # 0.0
            >>> print(positions['S2'])  # 60.0 (for 6-singer arrangement)
        """
        positions = {}
        for singer_id, position in self.singer_positions.items():
            angle = position.angular_position
            if angle is not None:
                positions[singer_id] = angle
        return positions

    @property
    def singer_count(self) -> int:
        """Get the number of actual singers in this performance (excluding empty positions).

        Returns:
            int: Number of singers.

        Example:
            >>> track = dataset.track('C3_14052025_S1_AGUACERO_FRAGMENT_P6_S5')
            >>> # 6 positions, but S5 is empty
            >>> print(track.singer_count)  # 5
        """
        return sum(
            1 for s in self.singer_positions.values() if not s.is_empty_position()
        )

    @property
    def singer_heights(self) -> List[str]:
        """Get raw singer height entries from CSV (split/stripped).

        Returns:
            list: List of height strings from CSV in original order.
        """
        raw = self._track_metadata.get("Singer_Height")
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]

    @property
    def singer_ids(self) -> List[str]:
        """Get list of all IDs (singers and positions) in order.

        Returns:
            list: List of IDs (e.g., ['S1', 'S2', 'S3', ...]) in the order
                they appear in the metadata.

        Example:
            >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
            >>> print(track.singer_ids)  # ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
            >>> # Use to iterate in order
            >>> for sid in track.singer_ids:
            ...     print(track.singer_positions[sid].gender)
        """
        return self._track_metadata.get("singer_ids", [])

    @property
    def singer_roles(self) -> List[str]:
        """Get raw singer role entries from CSV (split/stripped).

        Returns:
            list: List of role strings from CSV in original order.
        """
        raw = self._track_metadata.get("Singer_Roles")
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]

    @property
    def singer_positions(self) -> Dict[str, SingerPosition]:
        """Get per-position annotations as a dictionary keyed by ID.

        Returns:
            dict: Dictionary mapping IDs (e.g., "S1", "S2") to :class:`SingerPosition` objects.

        Example:
            >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
            >>> # Get all IDs
            >>> print(track.singer_ids)  # ['S1', 'S2', 'S3', 'S4', ...]
            >>> # Access a specific position's data
            >>> position = track.singer_positions['S1']
            >>> print(position.gender)  # 'M'
            >>> print(position.height)  # 172
            >>> # Iterate over all positions
            >>> for sid, position in track.singer_positions.items():
            ...     print(f"{sid}: {position.gender}")
        """
        return self._track_metadata.get("singers", {})

    @property
    def singers_facing_center(self) -> List[str]:
        """Get list of IDs facing toward the center/microphones.

        Returns:
            list: List of IDs (e.g., ['S1', 'S2', ...]) facing 'C' (Center).

        Example:
            >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
            >>> print(track.singers_facing_center)  # ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        """
        return [
            sid
            for sid in self.singer_ids
            if sid in self.singer_positions
            and self.singer_positions[sid].facing_direction == "C"
        ]

    @property
    def singers_facing_left(self) -> List[str]:
        """Get list of IDs facing counterclockwise (to their left).

        Returns:
            list: List of IDs facing 'L' (Left).
        """
        return [
            sid
            for sid in self.singer_ids
            if sid in self.singer_positions
            and self.singer_positions[sid].facing_direction == "L"
        ]

    @property
    def singers_facing_outside(self) -> List[str]:
        """Get list of IDs facing away from microphones (outward).

        Returns:
            list: List of IDs facing 'O' (Outside).
        """
        return [
            sid
            for sid in self.singer_ids
            if sid in self.singer_positions
            and self.singer_positions[sid].facing_direction == "O"
        ]

    @property
    def singers_facing_right(self) -> List[str]:
        """Get list of IDs facing clockwise (to their right).

        Returns:
            list: List of IDs facing 'R' (Right).
        """
        return [
            sid
            for sid in self.singer_ids
            if sid in self.singer_positions
            and self.singer_positions[sid].facing_direction == "R"
        ]

    @property
    def singing_group(self) -> Optional[str]:
        return self._track_metadata.get("Singing_Group")

    @property
    def song(self) -> Optional[str]:
        return self._track_metadata.get("Song")

    @property
    def unique_facing_directions(self) -> List[Optional[str]]:
        """Get list of unique facing directions present in this track.

        Returns:
            list: Sorted list of unique facing direction codes present.
                None values (empty positions) are included if present.

        Example:
            >>> track = dataset.track('C4_01042025_S1_VAMOSAREMAR_P17_S16')
            >>> print(track.unique_facing_directions)  # ['C', 'L', 'O', 'R']
        """
        unique = set(self.facing_directions)
        # Separate None from strings for proper sorting
        none_present = None in unique
        strings = sorted([x for x in unique if x is not None])
        return strings + ([None] if none_present else [])

    @property
    def audio_ortf_l(self) -> Tuple[np.ndarray, float]:
        """Load ORTF left channel audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        if self.audio_ortf_l_path is None:
            raise FileNotFoundError(
                "audio_ortf_l_path is None. Did you run .download()?"
            )
        return load_audio(self.audio_ortf_l_path)

    @property
    def audio_ortf_r(self) -> Tuple[np.ndarray, float]:
        """Load ORTF right channel audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        if self.audio_ortf_r_path is None:
            raise FileNotFoundError(
                "audio_ortf_r_path is None. Did you run .download()?"
            )
        return load_audio(self.audio_ortf_r_path)

    @property
    def audio_ortf_stereo(self) -> Tuple[np.ndarray, float]:
        """Load combined ORTF stereo audio (L and R channels)

        Returns:
            * np.ndarray - stereo audio signal (2 channels: [L, R])
            * float - sample rate
        """
        l_audio, l_sr = self.audio_ortf_l
        r_audio, r_sr = self.audio_ortf_r

        # Assume same sample rate (they're from the same recording)
        # Just ensure same length (trim to shorter if needed)
        min_len = min(len(l_audio), len(r_audio))
        stereo = np.stack([l_audio[:min_len], r_audio[:min_len]], axis=0)

        return stereo, float(l_sr)

    @property
    def audio_360(self) -> Optional[Tuple[np.ndarray, float]]:
        """Load 360° camera audio (stereo audio from Insta360 X3 action camera)

        This is the far-field audio captured by the 360° video camera's internal
        microphones. It serves as an "audience-perspective" reference recording
        at 48 kHz sample rate.

        Note: This property returns None for tracks from Session 1A (01/04/2025),
        as audio_360 files were not captured during that recording session.
        Approximately 84% of tracks (130/154) have audio_360 available.
        Only Session 1A tracks (01/04/2025) have no audio_360 available.

        To check if audio_360 is available, check:
        ``if track.audio_360 is not None:``

        Returns:
            * tuple or None: If available, returns tuple of:
                - np.ndarray - stereo audio signal (2 channels)
                - float - sample rate (typically 48000 Hz)
              If not available (audio_360_path is None), returns None.
        """
        if self.audio_360_path is None:
            return None
        return load_audio(self.audio_360_path)

    @core.cached_property
    def near_field_audio(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Load near-field recordings for all captured IDs.

        Returns:
            dict: dictionary mapping IDs to (audio, sample_rate) tuples
        """
        result = {}
        for singer_id, path in self.near_field_recordings.items():
            if path:
                result[singer_id] = load_audio(path)
        return result

    @property
    def available_nearfield_ids(self) -> List[str]:
        """IDs that have near-field recordings present in the index."""
        # Access _track_metadata to ensure metadata is loaded (raises exception if data_home doesn't exist)
        _ = self._track_metadata
        return list[str](self.near_field_recordings.keys())

    def near_field_audio_for(
        self, singer_id: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Get near-field audio for a single ID from the cached dict."""
        return self.near_field_audio.get(singer_id)

    def video(self, target_fps=None, frame_size=None):
        """Load 360° video data.

        Args:
            target_fps (float, optional): Target frames per second for resampling.
                If None, returns video at original fps.
            frame_size (tuple, optional): Target frame size as (height, width).
                If None, returns frames at original resolution.

        Returns:
            * np.ndarray - video array with shape (T, H, W, 3) in RGB format,
              where T is number of frames, H is height, W is width

        Raises:
            FileNotFoundError: If video_360_path is None or file cannot be opened

        Example:
            >>> track = dataset.track('C1_07052025_S1_MAMAINES_P6_S6')
            >>> # Load video at original resolution and fps
            >>> video_data = track.video()
            >>> # Load video with specific fps and frame size
            >>> video_data = track.video(target_fps=30, frame_size=(480, 640))
            >>> print(video_data.shape)  # (T, 480, 640, 3)
        """
        if self.video_360_path is None:
            raise FileNotFoundError("video_360_path is None. Did you run .download()?")
        return load_video(
            self.video_360_path, target_fps=target_fps, frame_size=frame_size
        )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The MULTIVOX dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="multivox",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata.csv")
        try:
            with open(metadata_path, "r") as fhandle:
                reader = csv.DictReader(fhandle)
                metadata = {}
                for row in reader:
                    track_id = row["Path"].strip()

                    # Parse other comma-separated fields
                    # Note: Facing_Direction column contains location codes (C, R, O, L, E)
                    # Locations_At_Circle is a count (6 or 16), same for all singers in a track
                    facing_directions = (
                        [s.strip() for s in row.get("Facing_Direction", "").split(",")]
                        if row.get("Facing_Direction")
                        else []
                    )
                    heights = (
                        [s.strip() for s in row.get("Singer_Height", "").split(",")]
                        if row.get("Singer_Height")
                        else []
                    )
                    genders = (
                        [s.strip() for s in row.get("Gender", "").split(",")]
                        if row.get("Gender")
                        else []
                    )
                    roles = (
                        [s.strip() for s in row.get("Singer_Roles", "").split(",")]
                        if row.get("Singer_Roles")
                        else []
                    )

                    # Locations_At_Circle is a single number (6 or 16)
                    locations_at_circle_str = row.get("Locations_At_Circle", "").strip()
                    locations_at_circle = None
                    if locations_at_circle_str and locations_at_circle_str.isdigit():
                        try:
                            locations_at_circle = int(locations_at_circle_str)
                        except ValueError:
                            locations_at_circle = None

                    # 1. Determine all singer IDs
                    # We want all positions in the circle (S1...SN)
                    # PLUS any extra IDs in Nearfield_Files_Captured
                    circle_ids = [f"S{i+1}" for i in range(locations_at_circle or 0)]

                    # Nearfield list might have S1, S2, S4... or Sinstructor
                    nearfield_list = row.get("Nearfield_Files_Captured", "").strip()
                    captured_ids = (
                        [s.strip() for s in nearfield_list.split(",")]
                        if nearfield_list
                        else []
                    )

                    # Extra IDs are those in captured but not in circle positions
                    # e.g. Sinstructor
                    circle_ids_set = set[str](circle_ids)
                    extra_ids = [
                        sid for sid in captured_ids if sid and sid not in circle_ids_set
                    ]

                    all_singer_ids = circle_ids + extra_ids

                    # 2. Create SingerPosition objects
                    singer_positions = {}
                    for i, singer_id in enumerate(all_singer_ids):
                        # Circle positions (0 to N-1)
                        is_circle = i < (locations_at_circle or 0)

                        height = None
                        gender = None
                        facing = None
                        role = None
                        if is_circle:
                            # Get metadata by index i
                            # (CSV columns are expected to have at least N values)
                            if (
                                i < len(heights)
                                and heights[i].strip()
                                and heights[i].strip() != "0"
                            ):
                                try:
                                    height = int(heights[i].strip())
                                except ValueError:
                                    height = None

                            gender = genders[i] if i < len(genders) else None
                            if gender == "E" or (gender and not gender.strip()):
                                gender = None

                            facing = (
                                facing_directions[i]
                                if i < len(facing_directions)
                                else None
                            )
                            if facing == "E" or (facing and not facing.strip()):
                                facing = None

                            role = roles[i] if i < len(roles) else None
                            if role == "E" or (role and not role.strip()):
                                role = None

                        singer_positions[singer_id] = SingerPosition(
                            singer_id=singer_id,
                            facing_direction=facing,
                            height=height,
                            gender=gender,
                            role=role,
                            locations_at_circle=(
                                locations_at_circle if is_circle else None
                            ),
                            index_at_circle=i if is_circle else None,
                        )

                    # Extract facing directions in order for convenience
                    # Include None values to represent empty positions
                    facing_directions_list = [
                        (
                            singer_positions[sid].facing_direction
                            if sid in singer_positions
                            else None
                        )
                        for sid in all_singer_ids
                    ]

                    # Store both raw row and parsed data
                    metadata[track_id] = {
                        **row,  # Keep all original fields
                        "singers": singer_positions,  # Add structured position data as dict
                        "singer_ids": all_singer_ids,  # All IDs (circle + extra)
                        "facing_directions": facing_directions_list,  # Pre-parsed list
                        "locations_at_circle": locations_at_circle,  # Layout geometry
                    }
                return metadata
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
