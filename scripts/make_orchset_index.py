import argparse
import os
import csv
import json

ORCHSET_INDEX_PATH = "../mir_dataset_loaders/indexes/orchset_index.json"


def make_orchset_index(data_path):

    predominant_inst_path = os.path.join(
        data_path, "Orchset - Predominant Melodic Instruments.csv")

    with open(predominant_inst_path, 'r') as fhandle:
        reader = csv.reader(fhandle, delimiter=',')
        raw_data = []
        for line in reader:
            if line[0] == 'excerpt':
                continue
            raw_data.append(line)

    tf_dict = {'TRUE': True, 'FALSE': False}

    index = {}
    for line in raw_data:
        track_id = line[0].split('.')[0]

        id_split = track_id.split('.')[0].split('-')
        if id_split[0] == 'Musorgski' or id_split[0] == 'Rimski':
            id_split[0] = '-'.join(id_split[:2])
            id_split.pop(1)

        melodic_instruments = [s.split(',') for s in line[1].split('+')]
        melodic_instruments = [item.lower() for sublist in melodic_instruments
                               for item in sublist]
        for i, inst in enumerate(melodic_instruments):
            if inst == 'string':
                melodic_instruments[i] = 'strings'
            elif inst == 'winds (solo)':
                melodic_instruments[i] = 'winds'
        melodic_instruments = list(set(melodic_instruments))

        index[track_id] = {
            'predominant_melodic_instruments-raw': line[1],
            'predominant_melodic_instruments-normalized': melodic_instruments,
            'alternating_melody': tf_dict[line[2]],
            'contains_winds': tf_dict[line[3]],
            'contains_strings': tf_dict[line[4]],
            'contains_brass': tf_dict[line[5]],
            'only_strings': tf_dict[line[6]],
            'only_winds': tf_dict[line[7]],
            'only_brass': tf_dict[line[8]],
            'composer': id_split[0],
            'work': '-'.join(id_split[1:-1]),
            'excerpt': id_split[-1][2:],
            'audio_path_stereo': 'Orchset/audio/stereo/{}.wav'.format(
                track_id),
            'audio_path_mono': 'Orchset/audio/mono/{}.wav'.format(track_id),
            'melody_path': 'Orchset/GT/{}.mel'.format(track_id)
        }

    with open(ORCHSET_INDEX_PATH, 'w') as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_orchset_index(args.orchset_data_path)


with __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Make Orchset index file.")
    PARSER.add_argument("orchset_data_path",
                        type=str,
                        help="Path to Orchset data folder.")

    main(PARSER.parse_args())
