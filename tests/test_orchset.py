# Created on 19 mar 2019 - 01:49:33 by mfuentes (mgfuenteslujambio@gmail.com)

from mir_datasets.download import orchset as orch_download
from mir_datasets.load import orchset as orch_load
import argparse

# ========== main ========

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Some testing on mir_datasets')
    parser.add_argument('path', type=str)
    parser.add_argument('melody_path', type=str)
    args =  parser.parse_args()

    # without indicating path
    orch_download.download_orchset()
    # indicating path
    orch_download.download_orchset(data_home=args.path)

    # load dataset
    orchset_data = orch_load.load_orchset(data_home=args.path)
    print('Load dataset ===')
    print(orchset_data.keys())
    print(orchset_data[list(orchset_data.keys())[0]])

    # load melody
    print('Load gth melody ===')
    melody_data = orch_load.load_orchset_melody(args.melody_path)
    # TODO: wouldn't be better to load by index?
    print(melody_data)

