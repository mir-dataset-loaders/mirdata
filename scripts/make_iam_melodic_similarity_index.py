import argparse
import hashlib
import json
import os
from mirdata.validate import md5

INDEX_PATH = '../mirdata/datasets/indexes/iam_melodic_similarity_index.json'
DATASET = 'MelodicSimilarityDataset'

def make_index(dataset_data_path):

    dataset_index = {
        'version': 1.0,
        'tracks': {},
    }
    idx = 0
    dataset_data_path_prev = dataset_data_path.split(DATASET)[0]

    for tradition in os.listdir(dataset_data_path):
        if '.' not in tradition:
            for song in os.listdir(os.path.join(dataset_data_path, tradition)):
                if '.' not in song:
                    # Declare track attributes
                    index = str(idx) + '_' + song.replace(' ', '_')
                    print(index)

                    # Audio
                    audio = (None, None)

                    # Section
                    sections = (None, None)
                    sections_finetuned = (None, None)

                    # Features
                    nyas = (None, None)
                    pitch = (None, None)
                    pitch_finetuned = (None, None)
                    tonic = (None, None)
                    tonic_finetuned = (None, None)

                    for file in os.listdir(os.path.join(dataset_data_path, tradition, song)):
                        # Audio
                        if '.mp3' in file or '.wav' in file:
                            audio_path = os.path.join(DATASET, tradition, song, file)
                            audio_checksum = md5(os.path.join(audio_path))
                            audio = (audio_path, audio_checksum)

                        # Sections
                        if file.endswith('.anot'):
                            sections_path = os.path.join(DATASET, tradition, song, file)
                            sections_checksum = md5(os.path.join(dataset_data_path_prev, sections_path))
                            sections = (sections_path, sections_checksum)

                        if '.anotEdit' in file:
                            sections_finetuned_path = os.path.join(DATASET, tradition, song, file)
                            sections_finetuned_checksum = md5(os.path.join(dataset_data_path_prev, sections_finetuned_path))
                            sections_finetuned = (sections_finetuned_path, sections_finetuned_checksum)

                        # Features
                        if '.flatSegNyas' in file:
                            nyas_path = os.path.join(DATASET, tradition, song, file)
                            nyas_checksum = md5(os.path.join(dataset_data_path_prev, nyas_path))
                            nyas = (nyas_path, nyas_checksum)

                        if file.endswith('.pitch') or file.endswith('.tpe'):
                            pitch_path = os.path.join(DATASET, tradition, song, file)
                            pitch_checksum = md5(os.path.join(dataset_data_path_prev, pitch_path))
                            pitch = (pitch_path, pitch_checksum)

                        elif '.pitch' in file or '.tpe' in file:
                            pitch_finetuned_path = os.path.join(DATASET, tradition, song, file)
                            pitch_finetuned_checksum = md5(os.path.join(dataset_data_path_prev, pitch_finetuned_path))
                            pitch_finetuned = (pitch_finetuned_path, pitch_finetuned_checksum)

                        if file.endswith('.tonic') or file.endswith('.tonic'):
                            tonic_path = os.path.join(DATASET, tradition, song, file)
                            tonic_checksum = md5(os.path.join(dataset_data_path_prev, tonic_path))
                            tonic = (tonic_path, tonic_checksum)

                        if file.endswith('.tonicFine') or file.endswith('.tonicFine'):
                            tonic_finetuned_path = os.path.join(DATASET, tradition, song, file)
                            tonic_finetuned_checksum = md5(os.path.join(dataset_data_path_prev, tonic_finetuned_path))
                            tonic_finetuned = (tonic_finetuned_path, tonic_finetuned_checksum)

                        dataset_index['tracks'][index] = {
                            'audio': audio,
                            'sections': sections,
                            'sections-finetuned': sections_finetuned,
                            'nyas': nyas,
                            'pitch': pitch,
                            'pitch-finetuned': pitch_finetuned,
                            'tonic': tonic,
                            'tonic-finetuned': tonic_finetuned,
                        }

                    idx = idx + 1

    with open(INDEX_PATH, 'w') as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_index(args.dataset_data_path)
    print("done!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make index file.')
    PARSER.add_argument(
        'dataset_data_path', type=str, help='Path to data folder.'
    )

    main(PARSER.parse_args())
