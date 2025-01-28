import argparse
import sys

def srt_to_txt(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    entries = content.strip().split('\n\n')
    output_lines = []

    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue
        text_lines = lines[2:]
        for line in text_lines:
            stripped_line = line.strip()
            if stripped_line:
                if not output_lines or stripped_line != output_lines[-1]:
                    output_lines.append(stripped_line)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
    except IOError:
        print(f"Error: Could not write to output file '{output_file}'")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SRT subtitle files to clean TXT files')
    parser.add_argument('-i', '--input', required=True, help='Input SRT file')
    parser.add_argument('-o', '--output', required=True, help='Output TXT file')
    args = parser.parse_args()
    
    srt_to_txt(args.input, args.output)