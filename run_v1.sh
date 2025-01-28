#!/bin/bash

set -e 

question=$1

# Create directories if they don't exist
rm -rf data/tmp_srt data/tmp_txt
mkdir -p data/tmp_srt data/tmp_txt

# Assuming scripts are in current directory (add ./ if needed)
bash ./download_yt_subtitles.sh -s "$question" -n 50 -o data/tmp_srt
bash ./convert_srt_to_txt.sh data/tmp_srt data/tmp_txt

# Use available Python 3 version if 3.12 isn't standard
python3.12 rag_v3.py \
  --input_folder data/tmp_txt \
  --question "$question" \
  --top_k 5 \
  --max_recursion 100

# Cleanup
rm -rf data/tmp_srt data/tmp_txt