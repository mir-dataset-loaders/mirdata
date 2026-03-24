import argparse
import glob
import json
import os
from mirdata.validate import md5

EGSET12_INDEX_PATH = "../mirdata/datasets/indexes/egset12_index_{}.json"

def make_egset12_index(egset12_data_path: str, version: str) -> None:
    #paths to files
    audio_files = glob.glob(os.path.join(egset12_data_path,"*.wav"))
    track_ids = sorted((os.path.basename(track_id)for track_id in audio_files)) 


    index_tracks = {}
    for track_id in track_ids:
        audio_checksum = md5(
            os.path.join(egset12_data_path, track_id),
        )
        annotation_checksum = md5(
            os.path.join(egset12_data_path, track_id.replace(".wav",".jams"))
        )
        index_tracks[track_id] = {
            "audio": (track_id, audio_checksum),
            "jams": (track_id.replace(".wav",".jams"),annotation_checksum), #annotations
        }
    
    egset12_index = {
        "version":version,
        "tracks": index_tracks,
        }
    
    with open(EGSET12_INDEX_PATH.format(version),"w")as fhandle:
        json.dump(egset12_index,fhandle,indent=2)

def main(args):
    make_egset12_index(args.egset12_data_path,args.version)
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make egset12 index file.")
    PARSER.add_argument("egset12_data_path",type=str, help="path to egset12 data folder.")
    PARSER.add_argument("version",type=str,help="index version")
    
    main(PARSER.parse_args())
    


