import jams_utils

def to_jams(track):
    """Convert the track's data into JAMS format.

    Args:
        track (object): A track object with the following attributes:
            - audio_path (str): Path to the audio file.
            - beats (annotations.BeatData): Beat annotations.
            - sections (annotations.SectionData): Section annotations.
            - chords (annotations.ChordData): Chord annotations.
            - key (annotations.KeyData): Key annotations.
            - title (str): Title of the track.

    Returns:
        jams.JAMS: The track's data in JAMS format.
    """
    return jams_utils.jams_converter(
        audio_path=track.audio_path,
        beat_data=[(track.beats, None)],
        section_data=[(track.sections, None)],
        chord_data=[(track.chords, None)],
        key_data=[(track.key, None)],
        metadata={"artist": "The Beatles", "title": track.title},
    )

# Example usage
track = ...  # load your track object here
jams = to_jams(track)
jams.save("example.jams")