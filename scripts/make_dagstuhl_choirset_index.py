import argparse
import glob
import hashlib
import json
import os
import string

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/dagstuhl_choirset_index.json"


NO_SCORE = [
    "Basses", "Overacted", "SE", "Solo", "Outtake"
]

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


def make_dataset_index(data_path):

    ### define directories

    audio_dir = os.path.join(data_path, 'audio_wav_22050_mono')

    index = {
        'version': 1.1,
        'tracks': {},
        'multitracks': {},
        'metadata': None
    }

    # define pieces directly from data directory
    pieces = sorted(list(set(["_".join(filename.split('/')[-1].split('_')[:4]) for filename in glob.glob(os.path.join(audio_dir, '*.wav'))])))

    for ip, piece in enumerate(pieces):

        index['multitracks'][piece] = {}

        ## add mixture audios

        # STM
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_Stereo_STM.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_stm"] = (
            "audio_wav_22050_mono/{}_Stereo_STM.wav".format(piece),
            audio_checksum
        )

        # STR
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_Stereo_STR.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_str"] = (
            "audio_wav_22050_mono/{}_Stereo_STR.wav".format(piece),
            audio_checksum
        )

        # STL
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_Stereo_STL.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_stl"] = (
            "audio_wav_22050_mono/{}_Stereo_STL.wav".format(piece),
            audio_checksum
        )

        # STL
        audio_mix_dir = os.path.join(
            data_path, "audio_wav_22050_mono", "{}_StereoReverb_STM.wav".format(piece)
        )
        audio_checksum = md5(audio_mix_dir)
        index["multitracks"][piece]["audio_rev"] = (
            "audio_wav_22050_mono/{}_StereoReverb_STM.wav".format(piece),
            audio_checksum
        )

        ## add each track inside the multitrack

        audio_files = sorted(
            glob.glob(os.path.join(audio_dir, '{}*.wav'.format(piece)))
        )

        singers = [singer.split('_')[-2] for singer in audio_files if not 'Stereo' in singer]

        # mics = [singer.split('_')[-1].split('.')[0] for singer in audio_files if not 'Stereo' in singer]
        # assert len(singers) == len(mics), "number of mics does not match number of singers for {}".format(piece)
        # set_singers = set(singers)

        index['multitracks'][piece]['tracks'] = []

        for sidx, singer in enumerate(singers):

            track_name = "{}_{}".format(piece, singer)
            index['tracks'][track_name] = {}


            index['multitracks'][piece]['tracks'].append(track_name)

            mics = [mic.split('_')[-1].split('.')[0] for mic in glob.glob(os.path.join(
                audio_dir, "{}_{}*.wav".format(piece, singer))
            )]

            ### add all fields for each track

            for mic in mics:

                ## add audio
                audio_stem_dir = os.path.join(
                    data_path, "audio_wav_22050_mono", "{}_{}_{}.wav".format(piece, singer, mic)
                )
                audio_checksum = md5(audio_stem_dir)

                index["tracks"][track_name]["audio_{}".format(mic.lower())] = (
                    "audio_wav_22050_mono/{}_{}_{}.wav".format(piece, singer, mic),
                    audio_checksum
                )

                ## add crepe f0s
                crepe_dir = os.path.join(
                    data_path, "annotations_csv_F0_CREPE", "{}_{}_{}.csv".format(piece, singer, mic)
                )
                crepe_checksum = md5(crepe_dir)

                index['tracks'][track_name]["f0_crepe_{}".format(mic.lower())] = (
                    "annotations_csv_F0_CREPE/{}_{}_{}.csv".format(piece, singer, mic),
                    crepe_checksum
                )

                ## add pyin f0s
                pyin_dir = os.path.join(
                    data_path, "annotations_csv_F0_PYIN", "{}_{}_{}.csv".format(piece, singer, mic)
                )
                pyin_checksum = md5(pyin_dir)

                index["tracks"][track_name]["f0_pyin_{}".format(mic.lower())] = (
                    "annotations_csv_F0_PYIN/{}_{}_{}.csv".format(piece, singer, mic),
                    pyin_checksum
                )

                ## add score when it exists

                # some have no associated score
                if not any(x in piece for x in NO_SCORE):

                    score_dir = os.path.join(
                        data_path, "annotations_csv_scorerepresentation", "{}_Stereo_STM_{}.csv".format(piece, singer[0])
                    )
                    score_checksum = md5(score_dir)

                    index["tracks"][track_name]["score"] = (
                        "annotations_csv_scorerepresentation/{}_Stereo_STM_{}.csv".format(piece, singer[0]),
                        score_checksum
                    )

            ## add beats for the full songs when available

            if not any(x in piece for x in NO_SCORE):

                ## add beats
                beats_dir = os.path.join(
                    data_path, "annotations_csv_beat", "{}_Stereo_STM.csv".format(piece)
                )
                beats_checksum = md5(beats_dir)

                index["multitracks"][piece]["beat"] = (
                    "annotations_csv_beat/{}_Stereo_STM.wav".format(piece),
                    beats_checksum
                )
        index['multitracks'][piece]['tracks'] = list(set(index['multitracks'][piece]['tracks']))


    ## add the manual annotations to their corresponding tracks
    manual_files = sorted(
        glob.glob(os.path.join(
            data_path, "annotations_csv_F0_manual", "*.csv")
        )
    )
    for mf in manual_files:
        track_name = "_".join(os.path.basename(mf).split('_')[:-1])

        manual_checksum = md5(mf)

        index["tracks"][track_name]["f0_manual"] = (
            "annotations_csv_F0_manual/{}".format(os.path.basename(mf)),
            manual_checksum
        )



    with open(DATASET_INDEX_PATH, 'w') as fhandle:
        json.dump(index, fhandle, indent=2)


# def main(args):
def main():
    #make_dataset_index(args.data_path)
    make_dataset_index('/Users/helenacuesta/Desktop/DagstuhlChoirSet')

#
# if __name__ == '__main__':
#     PARSER = argparse.ArgumentParser(description='Make Phenicx-anechoic index file.')
#     PARSER.add_argument(
#         'data_path', type=str, help='Path to Phenicx-anechoic data folder.'
#     )
#
#     main(PARSER.parse_args())

main()