#!/bin/bash -eux

MSD_FOLDER=/msd

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
