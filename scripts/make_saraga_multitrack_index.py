import argparse
import hashlib
import json
import os


SARAGA_INDEX_PATH = '../mirdata/datasets/indexes/saraga_multitrack_index.json'


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
    with open(file_path, 'rb') as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_saraga_multitrack_index(saraga_multitrack_data_path):

    saraga_index = {}
    idx = 0
    for root, dirs, files in os.walk(saraga_multitrack_data_path):
        for directory in dirs:  # Artists
            for root_, dirs_, files_ in os.walk(os.path.join(saraga_multitrack_data_path, directory)):
                for directory_ in dirs_:  # Works
                    for root__, dirs__, files__ in os.walk(os.path.join(saraga_multitrack_data_path, directory, directory_)):

                        # Declare track attributes
                        audio = (None, None)
                        audio_ghatam = (None, None)
                        audio_mridangam_left = (None, None)
                        audio_mridangam_right = (None, None)
                        audio_violin = (None, None)
                        audio_vocal = (None, None)
                        audio_vocal_s = (None, None)
                        ctonic = (None, None)
                        pitch = (None, None)
                        phrases = (None, None)
                        tempo = (None, None)
                        sama = (None, None)
                        sections = (None, None)
                        metadata = (None, None)

                        index = str(idx) + '_' + directory_
                        print(index)
                        for file in files__:
                            if '.mp3' in file:
                                if 'multitrack' in file:
                                    if 'ghatam' in file:
                                        audio_ghatam_path = os.path.join(directory, directory_, file)
                                        audio_ghatam_checksum = md5(os.path.join(saraga_multitrack_data_path, audio_ghatam_path))
                                        audio_ghatam = (os.path.join('saraga_multitrack/', audio_ghatam_path), audio_ghatam_checksum)
                                    if 'mridangam-left' in file:
                                        audio_mridangam_left_path = os.path.join(directory, directory_, file)
                                        audio_mridangam_left_checksum = md5(os.path.join(saraga_multitrack_data_path, audio_mridangam_left_path))
                                        audio_mridangam_left = (os.path.join('saraga_multitrack/', audio_mridangam_left_path), audio_mridangam_left_checksum)
                                    if 'mridangam-right' in file:
                                        mridangam_right_path = os.path.join(directory, directory_, file)
                                        mridangam_right_checksum = md5(os.path.join(saraga_multitrack_data_path, mridangam_right_path))
                                        audio_mridangam_right = (os.path.join('saraga_multitrack/', mridangam_right_path), mridangam_right_checksum)
                                    if 'violin' in file:
                                        audio_violin_path = os.path.join(directory, directory_, file)
                                        audio_violin_checksum = md5(os.path.join(saraga_multitrack_data_path, audio_violin_path))
                                        audio_violin = (os.path.join('saraga_multitrack/', audio_violin_path), audio_violin_checksum)
                                    if 'vocal-s' in file:
                                        audio_vocal_s_path = os.path.join(directory, directory_, file)
                                        audio_vocal_s_checksum = md5(os.path.join(saraga_multitrack_data_path, audio_vocal_s_path))
                                        audio_vocal_s = (os.path.join('saraga_multitrack/', audio_vocal_s_path), audio_vocal_s_checksum)
                                    if 'vocal' in file:
                                        audio_vocal_path = os.path.join(directory, directory_, file)
                                        audio_vocal_checksum = md5(os.path.join(saraga_multitrack_data_path, audio_vocal_path))
                                        audio_vocal = (os.path.join('saraga_multitrack/', audio_vocal_path), audio_vocal_checksum)

                                else:
                                    audio_path = os.path.join(directory, directory_, file)
                                    audio_checksum = md5(os.path.join(saraga_multitrack_data_path, audio_path))
                                    audio = (os.path.join('saraga_multitrack/', audio_path), audio_checksum)
                            if 'ctonic.txt' in file:
                                ctonic_path = os.path.join(directory, directory_, file)
                                ctonic_checksum = md5(os.path.join(saraga_multitrack_data_path, ctonic_path))
                                ctonic = (os.path.join('saraga_multitrack/', ctonic_path), ctonic_checksum)
                            if 'pitch.txt' in file:
                                pitch_path = os.path.join(directory, directory_, file)
                                pitch_checksum = md5(os.path.join(saraga_multitrack_data_path, pitch_path))
                                pitch = (os.path.join('saraga_multitrack/', pitch_path), pitch_checksum)
                            if 'tempo-manual.txt' in file:
                                tempo_path = os.path.join(directory, directory_, file)
                                tempo_checksum = md5(os.path.join(saraga_multitrack_data_path, tempo_path))
                                tempo = (os.path.join('saraga_multitrack/', tempo_path), tempo_checksum)
                            if 'sama-manual.txt' in file:
                                sama_path = os.path.join(directory, directory_, file)
                                sama_checksum = md5(os.path.join(saraga_multitrack_data_path, sama_path))
                                sama = (os.path.join('saraga_multitrack/', sama_path), sama_checksum)
                            if 'sections-manual-p.txt' in file:
                                sections_path = os.path.join(directory, directory_, file)
                                sections_checksum = md5(os.path.join(saraga_multitrack_data_path, sections_path))
                                sections = (os.path.join('saraga_multitrack/', sections_path), sections_checksum)
                            if 'mphrases-manual.txt' in file:
                                phrases_path = os.path.join(directory, directory_, file)
                                phrases_checksum = md5(os.path.join(saraga_multitrack_data_path, phrases_path))
                                phrases = (os.path.join('saraga_multitrack/', phrases_path), phrases_checksum)
                            if '.json' in file:
                                metadata_path = os.path.join(directory, directory_, file)
                                metadata_checksum = md5(os.path.join(saraga_multitrack_data_path, metadata_path))
                                metadata = (os.path.join('saraga_multitrack/', metadata_path), metadata_checksum)

                            saraga_index[index] = {
                                'audio': audio,
                                'audio-ghatam': audio_ghatam,
                                'audio-mridangam-left': audio_mridangam_left,
                                'audio-mridangam-right': audio_mridangam_right,
                                'audio-violin': audio_violin,
                                'audio-vocal-s': audio_vocal_s,
                                'audio-vocal': audio_vocal,
                                'ctonic': ctonic,
                                'pitch': pitch,
                                'phrases': phrases,
                                'tempo': tempo,
                                'sama': sama,
                                'sections': sections,
                                'metadata': metadata
                            }

                        idx += 1  # In case there is two or more records called the same

    with open(SARAGA_INDEX_PATH, 'w') as fhandle:
        json.dump(saraga_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_saraga_multitrack_index(args.saraga_multitrack_data_path)
    print("done!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make Saraga Multitrack index file.')
    PARSER.add_argument(
        'saraga_multitrack_data_path', type=str, help='Path to saraga_multitrack data folder.'
    )

    main(PARSER.parse_args())
