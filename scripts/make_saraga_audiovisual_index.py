import argparse
import hashlib
import json
import os
from mirdata.validate import md5


SARAGA_CARNATIC_INDEX_PATH = '../mirdata/datasets/indexes/saraga_audiovisual_index.json'
DATASET = 'saraga'

def make_saraga_audiovisual_index(dataset_data_path):

    saraga_index = {
        'version': 1.0,
        'tracks': {},
    }
    idx = 0
    dataset_data_path_prev = dataset_data_path.split(DATASET)[0]
    for concert in os.listdir(dataset_data_path):
        if '.' not in concert:
            for song in os.listdir(os.path.join(dataset_data_path, concert)):
                if '.' not in song:
                    # Declare track attributes
                    index = str(idx) + '_' + song.replace(' ', '_')
                    print(index)

                    # Audio
                    audio = (None, None)
                    audio_mridangam_left = (None, None)
                    audio_mridangam_right = (None, None)
                    audio_violin = (None, None)
                    audio_vocal = (None, None)

                    # Video
                    video = (None, None)

                    # Gesture
                    keypoints_mridangam = (None, None)
                    keypoints_violin = (None, None)
                    keypoints_singer = (None, None)
                    scores_mridangam = (None, None)
                    scores_violin = (None, None)
                    scores_singer = (None, None)

                    # Metadata
                    metadata = (None, None)

                    for file in os.listdir(os.path.join(dataset_data_path, concert, song)):
                        # Audio
                        if '.wav' in file:
                            if file == song + '.wav':
                                audio_path = os.path.join(DATASET, concert, song, file)
                                audio_checksum = md5(os.path.join(dataset_data_path_prev, audio_path))
                                audio = (audio_path, audio_checksum)
                            if 'mridangam-left' in file:
                                audio_mridangam_left_path = os.path.join(DATASET, concert, song, file)
                                audio_mridangam_left_checksum = md5(os.path.join(dataset_data_path_prev, audio_mridangam_left_path))
                                audio_mridangam_left = (audio_mridangam_left_path, audio_mridangam_left_checksum)
                            if 'mridangam-right' in file:
                                mridangam_right_path = os.path.join(DATASET, concert, song, file)
                                mridangam_right_checksum = md5(os.path.join(dataset_data_path_prev, mridangam_right_path))
                                audio_mridangam_right = (mridangam_right_path, mridangam_right_checksum)
                            if 'violin' in file:
                                audio_violin_path = os.path.join(DATASET, concert, song, file)
                                audio_violin_checksum = md5(os.path.join(dataset_data_path_prev, audio_violin_path))
                                audio_violin = (audio_violin_path, audio_violin_checksum)
                            if 'vocal' in file:
                                audio_vocal_path = os.path.join(DATASET, concert, song, file)
                                audio_vocal_checksum = md5(os.path.join(dataset_data_path_prev, audio_vocal_path))
                                audio_vocal = (audio_vocal_path, audio_vocal_checksum)

                        # Video
                        if '.mov' in file:
                            video_path = os.path.join(DATASET, concert, song, file)
                            video_checksum = md5(os.path.join(dataset_data_path_prev, video_path))
                            video = (video_path, video_checksum)

                        # Gesture
                        if file == 'mridangam':
                            for gesture_file in os.listdir(os.path.join(dataset_data_path, concert, song, file)):
                                if 'kpts.npy' in gesture_file:
                                    keypoints_mridangam_path = os.path.join(DATASET, concert, song, file, gesture_file)
                                    keypoints_mridangam_checksum = md5(os.path.join(dataset_data_path_prev, keypoints_mridangam_path))
                                    keypoints_mridangam = (keypoints_mridangam_path, keypoints_mridangam_checksum)
                                if 'scores.npy' in gesture_file:
                                    scores_mridangam_path = os.path.join(DATASET, concert, song, file, gesture_file)
                                    scores_mridangam_checksum = md5(os.path.join(dataset_data_path_prev, scores_mridangam_path))
                                    scores_mridangam = (scores_mridangam_path, scores_mridangam_checksum)
                        if file == 'singer':
                            for gesture_file in os.listdir(os.path.join(dataset_data_path, concert, song, file)):
                                if 'kpts.npy' in gesture_file:
                                    keypoints_singer_path = os.path.join(DATASET, concert, song, file, gesture_file)
                                    keypoints_singer_checksum = md5(os.path.join(dataset_data_path_prev, keypoints_singer_path))
                                    keypoints_singer = (keypoints_singer_path, keypoints_singer_checksum)
                                if 'scores.npy' in gesture_file:
                                    scores_singer_path = os.path.join(DATASET, concert, song, file, gesture_file)
                                    scores_singer_checksum = md5(os.path.join(dataset_data_path_prev, scores_singer_path))
                                    scores_singer = (scores_singer_path, scores_singer_checksum)
                        if file == 'violin':
                            for gesture_file in os.listdir(os.path.join(dataset_data_path, concert, song, file)):
                                if 'kpts.npy' in gesture_file:
                                    keypoints_violin_path = os.path.join(DATASET, concert, song, file, gesture_file)
                                    keypoints_violin_checksum = md5(os.path.join(dataset_data_path_prev, keypoints_violin_path))
                                    keypoints_violin = (keypoints_violin_path, keypoints_violin_checksum)
                                if 'scores.npy' in gesture_file:
                                    scores_violin_path = os.path.join(DATASET, concert, song, file, gesture_file)
                                    scores_violin_checksum = md5(os.path.join(dataset_data_path_prev, scores_violin_path))
                                    scores_violin = (scores_violin_path, scores_violin_checksum)

                        # Metadata
                        if '.json' in file and not file.startswith('._'):
                            metadata_path = os.path.join(DATASET, concert, song, file)
                            metadata_checksum = md5(os.path.join(dataset_data_path_prev, metadata_path))
                            metadata = (metadata_path, metadata_checksum)

                        saraga_index['tracks'][index] = {
                            'audio-mix': audio,
                            'video': video,
                            'audio-mridangam-left': audio_mridangam_left,
                            'audio-mridangam-right': audio_mridangam_right,
                            'audio-violin': audio_violin,
                            'audio-vocal': audio_vocal,
                            'keypoints-mridangam': keypoints_mridangam,
                            'keypoints-singer': keypoints_singer,
                            'keypoints-violin': keypoints_violin,
                            'scores-mridangam': scores_mridangam,
                            'scores-singer': scores_singer,
                            'scores-violin': scores_violin,
                            'metadata': metadata,
                        }

                    idx = idx + 1

    with open(SARAGA_CARNATIC_INDEX_PATH, 'w') as fhandle:
        json.dump(saraga_index, fhandle, indent=2)


def main(args):
    print("creating index...")
    make_saraga_audiovisual_index(args.dataset_data_path)
    print("done!")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make Saraga Carnatic index file.')
    PARSER.add_argument(
        'dataset_data_path', type=str, help='Path to Saraga Carnatic data folder.'
    )

    main(PARSER.parse_args())
