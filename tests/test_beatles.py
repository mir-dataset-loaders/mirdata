# Created on 02 abr 2019 - 05:28:09 by mfuentes (mgfuenteslujambio@gmail.com)

from mir_dataset_loaders import beatles
import argparse

# ========== main ========

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Some testing on mir_datasets')
    parser.add_argument('path', type=str)
    args =  parser.parse_args()

    # without indicating path
    # orch_download.download_orchset()
    # indicating path
    # orch_download.download_orchset(data_home=args.path)

    # load dataset
    beatles_data = beatles.load(data_home=args.path)
    print('Load dataset ===')
    print(beatles_data.keys())
    print(beatles_data[list(beatles_data.keys())[0]])  # both annotators
    print(beatles_data[beatles_data.keys()[1]])