import argparse
import csv
import hashlib
import json
import os
import librosa
import soundfile as sf

RWC_JAZZ_INDEX_PATH = "../mirdata/datasets/indexes/rwc_jazz_index_2.0.json"


def md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_rwc_jazz_index(data_path):
    dataset_root = os.path.join(data_path, "rwc_jazz")
    metadata_file = os.path.join(
        dataset_root,
        "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
        "metadata.csv",
    )

    with open(metadata_file, "r", encoding="utf-8") as fhandle:
        reader = csv.DictReader(fhandle, delimiter=";")
        piece = []
        for row in reader:
            rwcid = (row.get("RWCID") or "").strip()
            if not rwcid.startswith("RWC_J"):
                continue

            piece.append(rwcid[-3:])

    track_ids = sorted([f"RWC_J{p}" for p in piece])

    rwc_jazz_index = {"version": "2.0", "tracks": {}, "metadata": {}}
    audio_records = []
    for track_id in track_ids:
        pid = track_id[-3:]
        rm_id = f"RM-J{pid}"

        audio_rel = os.path.join("RWC-J", f"{track_id}.wav")
        audio_abs = os.path.join(dataset_root, audio_rel)
        if not os.path.exists(audio_abs):
            raise FileNotFoundError(f"Audio file not found: {audio_abs}")
        audio_rel = os.path.relpath(audio_abs, dataset_root).replace("\\", "/")
        audio_checksum = md5(audio_abs)

        length, sr, channels = audio_features(audio_abs)
        audio_record = {
            "duration_seconds": round(length, 3),
            "sample_rate": sr,
            "num_channels": channels,
        }
        audio_records.append(audio_record)




        beats_rel = os.path.join(
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
            "01_annotations_preprocessed",
            "beats",
            "RWC-J",
            f"{track_id}.csv",
        )
        beats_abs = os.path.join(dataset_root, beats_rel)
        if os.path.exists(beats_abs):
            beats_idx = beats_rel.replace("\\", "/")
            beats_md5 = md5(beats_abs)
        else:
            beats_idx, beats_md5 = None, None

        midi_rel = os.path.join(
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
            "01_annotations_preprocessed",
            "MIDI_aligned",
            "RWC-J",
            f"{track_id}.mid",
        )
        midi_abs = os.path.join(dataset_root, midi_rel)
        if os.path.exists(midi_abs):
            midi_idx = midi_rel.replace("\\", "/")
            midi_md5 = md5(midi_abs)
        else:
            midi_idx, midi_md5 = None, None

        rwc_jazz_index["tracks"][track_id] = {
            "audio": (audio_rel, audio_checksum),
            "beats": (beats_idx, beats_md5),
            "midi": (midi_idx, midi_md5),
            "audio_features": audio_record,
        }

    metadata_rel = os.path.relpath(metadata_file, dataset_root).replace("\\", "/")
    rwc_jazz_index["metadata"]["rwc-jazz-metadata"] = (
        metadata_rel,
        md5(metadata_file),
    )

    with open(RWC_JAZZ_INDEX_PATH, "w", encoding="utf-8") as fhandle:
        json.dump(rwc_jazz_index, fhandle, indent=2)

    print(f"Wrote {len(track_ids)} tracks to {RWC_JAZZ_INDEX_PATH}")

def audio_features(file_path):
    try: #faster way of reading the header of the file if exists instead of loading the whole file
        info = sf.info(file_path)
        return info.duration, info.samplerate, info.channels
    except Exception: #in case the file is not supported by soundfile, use librosa
        y, sr = librosa.load(file_path, sr=None, mono=False)
        num_channels = 1 if y.ndim == 1 else y.shape[0]
        length = y.shape[-1] / sr
    return length, sr, num_channels


def main(args):
    make_rwc_jazz_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make RWC-Jazz index file.")
    PARSER.add_argument(
        "--data_path",
        type=str,
        help="Path to parent directory of rwc_jazz.",
    )
    main(PARSER.parse_args())
