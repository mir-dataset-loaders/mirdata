#!/bin/bash -eux

# Description:
#
# make_msd_index.sh picks a (deterministically) random 10,000
# tracks from the Million Song Dataset and computes MD5 hashes.

# Usage:
#
# ./make_msd_index.sh <msd_folder>

# Dependencies:
#
# * jq (installation instructions: https://stedolan.github.io/jq/download/)

MSD_FOLDER=$1

get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

find $MSD_FOLDER -name *.h5 > msd-index.txt 2>/dev/null

shuf msd-index.txt --random-source=<(get_seeded_random 42) | head -n10000 > msd-sample.txt

cat msd-sample.txt | while read f; do
    md5sum $f >> md5sums.txt
done

cat md5sums.txt |
    jq --raw-input '. | split("  ") | {(.[1] | split("/")[-1] | split(".")[0]): {data: [(.[1] | sub("/msd/"; "")), (.[0])]}}' > msd_index.json
