# Created on 02 abr 2019 - 05:28:09 by mfuentes (mgfuenteslujambio@gmail.com)

from mir_dataset_loaders import salami
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
    salami_data = salami.load(data_home=args.path)
    print('Load dataset ===')
    print(salami_data.keys())
    print(salami_data[list(salami_data.keys())[0]])  # both annotators
    print(salami_data['2'])  # only one annotator
    print(salami_data['2'][1][0][0])
    print(salami_data['2'][1][0][1])
    print(salami_data['4'][1][0][2])
    print(salami_data['55'][3])
