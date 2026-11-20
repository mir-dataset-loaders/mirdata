import argparse
import glob
import json
import os
import librosa
from mirdata.validate import md5
import soundfile as sf

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/dataset_index.json"


def make_dataset_index(dataset_data_path):
    annotation_dir = os.path.join(dataset_data_path, "annotation")
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.lab"))
    track_ids = sorted([os.path.basename(f).split(".")[0] for f in annotation_files])

    total_dataset_bytes = 0
    # top-key level metadata
    metadata_checksum = md5(os.path.join(dataset_data_path, "id_mapping.txt"))
    total_dataset_bytes += get_file_size(
        os.path.join(dataset_data_path, "id_mapping.txt")
    )
    index_metadata = {"metadata": {"id_mapping": ("id_mapping.txt", metadata_checksum)}}
    # top-key level tracks
    index_tracks = {}
    audio_records = []
    for track_id in track_ids:
        # getting audio checksum and size
        audio_full_path = os.path.join(
            dataset_data_path, "Wavfile/{}.wav".format(track_id)
        )
        audio_checksum = md5(audio_full_path)
        audio_size = get_file_size(audio_full_path)
        total_dataset_bytes += audio_size

        # getting annotation checksum and size
        annotation_full_path = os.path.join(
            dataset_data_path, "annotation/{}.lab".format(track_id)
        )
        annotation_checksum = md5(annotation_full_path)
        annotation_size = get_file_size(annotation_full_path)
        total_dataset_bytes += annotation_size

        length, sr, channels = audio_features(audio_full_path)
        audio_record = {
            "duration_seconds": round(length, 3),
            "sample_rate": sr,
            "num_channels": channels,
        }
        audio_records.append(audio_record)

        index_tracks[track_id] = {
            "audio": ("Wavfile/{}.wav".format(track_id), audio_checksum),
            "annotation": ("annotation/{}.lab".format(track_id), annotation_checksum),
            "audio_features": {
                "audio": audio_record,
            },
        }

    # If your index has multitrack-level audio files (such as a mix), add
    # audio features for those files too. This assumes index_multitracks has already
    # been built; Entries with no multitrack-level audio path (only a "tracks" list,
    # for example) are kept as-is; no audio_features are added to them.

    # top-key level version
    dataset_index = {
        "version": None,
        "dataset_size_bytes": total_dataset_bytes,
    }

    # combine all in dataset index
    dataset_index.update(index_metadata)
    dataset_index.update({"tracks": index_tracks})

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def audio_features(file_path):
    try:  # faster way of reading the header of the file if exists instead of loading the whole file
        info = sf.info(file_path)
        return info.duration, info.samplerate, info.channels
    except Exception:  # in case the file is not supported by soundfile, use librosa
        y, sr = librosa.load(file_path, sr=None, mono=False)
        num_channels = 1 if y.ndim == 1 else y.shape[0]
        length = y.shape[-1] / sr
    return length, sr, num_channels


def get_file_size(file_path):
    """Return file size in bytes."""
    return os.path.getsize(file_path)


def main(args):
    make_dataset_index(args.dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to dataset data folder."
    )

    main(PARSER.parse_args())
