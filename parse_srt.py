import json
import re
import argparse

def srt_to_json(srt_file_path, json_file_path):
    entries = []
    
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        try:
            # Parse time codes
            time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
            if not time_match:
                continue
            
            start_time = time_match.group(1)
            end_time = time_match.group(2)
            
            # Combine text lines and clean up
            text = ' '.join(line.strip() for line in lines[2:] if line.strip())
            
            entries.append({
                'start_time': start_time,
                'end_time': end_time,
                'text': text
            })
        except (IndexError, ValueError):
            continue
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(entries, json_file, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SRT subtitle files to JSON format')
    parser.add_argument('input', help='Input SRT file path')
    parser.add_argument('-o', '--output', help='Output JSON file path', default='output.json')
    
    args = parser.parse_args()
    
    srt_to_json(args.input, args.output)
    print(f'Successfully converted {args.input} to {args.output}')