import sys, json

_, input_path, output_path = sys.argv

header, *lines = open(input_path).readlines()
fout = open(output_path, 'w')

fout.write(header)

min_gap = 0.1
max_gap = 1.0

prev_offset = None
prev_text = None
running_shortcut = 0.

for line in lines:
    offset, mode, text = json.loads(line)
    if prev_offset is None:
        prev_offset = offset
        prev_text = text
    elif offset - prev_offset < min_gap:
        prev_text += text
    else:
        fout.write(json.dumps([prev_offset - running_shortcut, 'o', prev_text]) + '\n')

        if offset - prev_offset > max_gap:
            running_shortcut += (offset - prev_offset) - max_gap

        prev_offset = offset
        prev_text = text

if prev_offset is not None:
    fout.write(json.dumps([prev_offset - running_shortcut, 'o', prev_text]) + '\n')

