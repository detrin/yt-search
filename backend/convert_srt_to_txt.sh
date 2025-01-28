#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process all SRT files in input directory
for srt_file in "$INPUT_DIR"/*.srt; do
    if [ -f "$srt_file" ]; then
        base_name=$(basename "$srt_file" .srt)
        txt_file="$OUTPUT_DIR/$base_name.txt"
        echo "Converting $srt_file to $txt_file"
        python3 srt_to_txt.py -i "$srt_file" -o "$txt_file"
    fi
done

echo "Conversion complete. Files saved to $OUTPUT_DIR"