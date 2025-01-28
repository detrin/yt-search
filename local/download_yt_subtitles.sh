#!/bin/bash

# Default values
search_phrase="kubernetes"
num_results=10
output_dir=""

# Parse command line options
while getopts ":s:n:o:" opt; do
  case $opt in
    s) search_phrase="$OPTARG";;
    n) num_results="$OPTARG";;
    o) output_dir="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Create output directory if not specified
if [ -z "$output_dir" ]; then
  output_dir="/tmp/$(uuidgen)"
  mkdir -p "$output_dir"
  echo "Created temporary directory: $output_dir"
else
  mkdir -p "$output_dir"
fi

# Download auto-generated subtitles
yt-dlp "ytsearch${num_results}:${search_phrase}" \
  --write-auto-subs \
  --sub-lang en \
  --convert-subs srt \
  --skip-download \
  -P "home:$output_dir"

# Download regular subtitles
yt-dlp "ytsearch${num_results}:${search_phrase}" \
  --write-subs \
  --sub-lang en \
  --convert-subs srt \
  --skip-download \
  -P "home:$output_dir"

echo "Subtitles downloaded to: $output_dir"