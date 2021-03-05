import argparse
import json
import os
from mirdata.validate import md5
from mirdata.download_utils import RemoteFileMetadata
from numpy import save
import gdown
import pyarrow.parquet as pq
from tqdm import tqdm

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/4mula_tiny_index.json"


def make_4mula_tiny_index(dataset_data_path):

    annotation_dir = os.path.join(dataset_data_path["dataset_folder"], "annotation")
    melspectrogram_dir = os.path.join(dataset_data_path["dataset_folder"], "melspectrogram")

    if not os.path.exists(annotation_dir):
        os.mkdir(annotation_dir)
    if not os.path.exists(melspectrogram_dir):
        os.mkdir(melspectrogram_dir)

    batch_size = 1
    dataset = pq.ParquetFile(f"{dataset_data_path['dataset_folder']}/{dataset_data_path['filename']}")
    batches = dataset.iter_batches(batch_size)  # batches will be a generator

    # top-key level tracks
    index_tracks = {}
    for track in tqdm(batches):

        with open(f"{annotation_dir}/{track.column('music_id')[0]}.csv", 'w') as f:
            f.write(
                'music_id,music_name,music_lang,'
                'music_lyrics,art_id,art_name,'
                'art_rank,main_genre,related_genre,'
                'related_art,related_music,musicnn_tags\n')
            f.write(
                f"{track.column('music_id')[0]},{track.column('music_name')[0]},{track.column('music_lang')[0]},"
                f"{track.column('music_lyrics')[0]},{track.column('art_id')[0]},{track.column('art_name')[0]},"
                f"{track.column('art_rank')[0]},{track.column('main_genre')[0]},{track.column('related_genre')[0]},"
                f"{track.column('related_art')[0]},{track.column('related_music')[0]},{track.column('musicnn_tags')[0]}")

        with open(f"{melspectrogram_dir}/{track.column('music_id')[0]}.npy", 'wb') as f:
            save(f, track.column('melspectrogram')[0])

        audio_checksum = md5(
            os.path.join(dataset_data_path["dataset_folder"], f"melspectrogram/{track.column('music_id')[0]}.npy")
        )
        annotation_checksum = md5(
            os.path.join(dataset_data_path["dataset_folder"], f"annotation/{track.column('music_id')[0]}.csv")
        )

        index_tracks[track.column('music_id')[0]] = {
            "melspectrogram": (f"melspectrogram/{track.column('music_id')[0]}.npy", audio_checksum),
            "annotation": (f"annotation/{track.column('music_id')[0]}.csv", annotation_checksum)
        }


    # top-key level metadata

    index_metadata = {"metadata": None}

    # top-key level version
    dataset_index = {"version": "1.0"}

    # combine all in dataset index
    dataset_index.update(index_metadata)
    dataset_index.update({"tracks": index_tracks})

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def download_gdrive_file(remote: RemoteFileMetadata, quiet: bool = False):
    """Download files from Google Drive

    :param remote: RemoteFileMetadata instance that contains url, filename, and destination_dir as attributes
    :param quiet: Quiet progress bar from gdown
    :return: downloaded file
    """
    dataset_path = os.path.basename(remote.destination_dir)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    output = f"{dataset_path}/{remote.filename}"
    if not os.path.exists(output):
        gdown.download(url=remote.url, output=output, quiet=quiet)
    return output


def main(dataset_data_path: dict):
    make_4mula_tiny_index(dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make 4mula tiny index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to 4mula tiny data folder.", default="downloads"
    )
    args = PARSER.parse_args()

    filename = "4mula_tiny.parquet"
    destination_dir = "downloads" if len(args.dataset_data_path) is 0 else args.dataset_data_path

    remote = RemoteFileMetadata(url='https://drive.google.com/uc?id=1ZBOpMarkVD38M-eubaZXzWyHXaHspsNV',
                                filename=f"{filename}", destination_dir=destination_dir, checksum=None)

    dataset_data_path = download_gdrive_file(remote)
    args = {"dataset_folder": destination_dir, "filename": filename}
    main(args)
