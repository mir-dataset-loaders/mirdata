import argparse
import hashlib
import collections
import json
import os


SARAGA_INDEX_PATH = '../mirdata/indexes/saraga_index.json'


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


def make_saraga_index(saraga_data_path):

    saraga_index = {}
    for root, dirs, files in os.walk(saraga_data_path):
        for directory in dirs:  # Hindustani vs. Carnatic
            for root_, dirs_, files_ in os.walk(os.path.join(saraga_data_path, directory)):
                for directory_ in dirs_:  # IDs
                    for root__, dirs__, files__ in os.walk(os.path.join(saraga_data_path, directory, directory_)):

                        # Declare track attributes
                        audio = (None, None)
                        ctonic = (None, None)
                        pitch = (None, None)
                        pitch_v = (None, None)
                        bpm = (None, None)
                        tempo = (None, None)
                        sama = (None, None)
                        sections = (None, None)
                        phrases = (None, None)
                        metadata = (None, None)

                        for file in files__:
                            index = str(directory) + '_' + directory_
                            if '.mp3' in file:
                                audio_path = os.path.join(directory, directory_, file)
                                audio_checksum = md5(os.path.join(saraga_data_path, audio_path))
                                audio = (os.path.join('saraga1.0/', audio_path), audio_checksum)
                            if 'ctonic' in file:
                                ctonic_path = os.path.join(directory, directory_, file)
                                ctonic_checksum = md5(os.path.join(saraga_data_path, ctonic_path))
                                ctonic = (os.path.join('saraga1.0/', ctonic_path), ctonic_checksum)
                            if 'pitch.' in file:
                                pitch_path = os.path.join(directory, directory_, file)
                                pitch_checksum = md5(os.path.join(saraga_data_path, pitch_path))
                                pitch = (os.path.join('saraga1.0/', pitch_path), pitch_checksum)
                            if 'pitch-vocal' in file:
                                pitch_v_path = os.path.join(directory, directory_, file)
                                pitch_v_checksum = md5(os.path.join(saraga_data_path, pitch_v_path))
                                pitch_v = (os.path.join('saraga1.0/', pitch_v_path), pitch_v_checksum)
                            if 'bpm' in file:
                                bpm_path = os.path.join(directory, directory_, file)
                                bpm_checksum = md5(os.path.join(saraga_data_path, bpm_path))
                                bpm = (os.path.join('saraga1.0/', bpm_path), bpm_checksum)
                            if 'tempo' in file:
                                tempo_path = os.path.join(directory, directory_, file)
                                tempo_checksum = md5(os.path.join(saraga_data_path, tempo_path))
                                tempo = (os.path.join('saraga1.0/', tempo_path), tempo_checksum)
                            if 'sama' in file:
                                sama_path = os.path.join(directory, directory_, file)
                                sama_checksum = md5(os.path.join(saraga_data_path, sama_path))
                                sama = (os.path.join('saraga1.0/', sama_path), sama_checksum)
                            if 'sections-manual.txt' in file:
                                sections_path = os.path.join(directory, directory_, file)
                                sections_checksum = md5(os.path.join(saraga_data_path, sections_path))
                                sections = (os.path.join('saraga1.0/', sections_path), sections_checksum)
                            if 'phrase' in file:
                                phrases_path = os.path.join(directory, directory_, file)
                                phrases_checksum = md5(os.path.join(saraga_data_path, phrases_path))
                                phrases = (os.path.join('saraga1.0/', phrases_path), phrases_checksum)
                            if '.json' in file:
                                metadata_path = os.path.join(directory, directory_, file)
                                metadata_checksum = md5(os.path.join(saraga_data_path, metadata_path))
                                metadata = (os.path.join('saraga1.0/', metadata_path), metadata_checksum)

                            saraga_index[index] = {
                                'audio': audio,
                                'ctonic': ctonic,
                                'pitch': pitch,
                                'pitch_vocal': pitch_v,
                                'bpm': bpm,
                                'tempo': tempo,
                                'sama': sama,
                                'sections': sections,
                                'phrases': phrases,
                                'metadata': metadata
                            }

    with open(SARAGA_INDEX_PATH, 'w') as fhandle:
        json.dump(saraga_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_saraga_index(args.saraga_data_path)
    print("done!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make Saraga index file.')
    PARSER.add_argument(
        'saraga_data_path', type=str, help='Path to Saraga data folder.'
    )

    main(PARSER.parse_args())
