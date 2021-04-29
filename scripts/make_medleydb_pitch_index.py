import argparse
import hashlib
import json
import os


MEDLEYDB_PITCH_INDEX_PATH = "mirdata/datasets/indexes/medleydb_pitch_index_{}.json"


def md5(file_path):
    """Get md5 hash of a file.

    Parameters
    ----------
    file_path: str
        File path.

    Returns
    -------
    md5_hash: str
        md5 hash of data in file_path
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def strip_first_dir(full_path):
    return os.path.join(*(full_path.split(os.path.sep)[1:]))


def make_medleydb_pitch_index(data_path, version):
    assert version in [2.0, 3.0], "invalid version"

    metadata_path = os.path.join(
        data_path, "MedleyDB-Pitch", "medleydb_pitch_metadata.json"
    )
    with open(metadata_path, "r") as fhandle:
        metadata = json.load(fhandle)

    pitch_index = {}
    pitch_index["tracks"] = {}
    for trackid in metadata.keys():
        audio_path = os.path.join(data_path, metadata[trackid]["audio_path"])
        audio_checksum = md5(audio_path)
        local_pitch_path = os.path.join(data_path, metadata[trackid]["pitch_path"])
        pitch_checksum = md5(local_pitch_path)

        fullid = os.path.basename(audio_path).split(".")[0]
        pitch_index["tracks"][fullid] = {
            "audio": (strip_first_dir(metadata[trackid]["audio_path"]), audio_checksum),
            "pitch": (strip_first_dir(metadata[trackid]["pitch_path"]), pitch_checksum),
        }
        if version == 3.0:
            note_path = "medleydb-pitch-pyin-notes/{}_vamp_pyin_pyin_notes.csv".format(
                fullid
            )
            pitch_index["tracks"][fullid]["notes_pyin"] = (
                note_path,
                md5(os.path.join(data_path)),
            )
    pitch_index["version"] = version
    pitch_index["metadata"] = {
        "medleydb_pitch_metadata": ("medleydb_pitch_metadata.json", md5(metadata_path))
    }

    with open(MEDLEYDB_PITCH_INDEX_PATH.format(version), "w") as fhandle:
        json.dump(pitch_index, fhandle, indent=2)


def main(args):
    make_medleydb_pitch_index(args.mdb_pitch_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make MedleyDB-Pitch index file.")
    PARSER.add_argument(
        "mdb_pitch_data_path", type=str, help="Path to MedleyDB-Pitch data folder."
    )
    PARSER.add_argument("version", type=str, help="index version.")

    main(PARSER.parse_args())
