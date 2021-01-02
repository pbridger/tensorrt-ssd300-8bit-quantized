import sys, json

_, input_path, output_path = sys.argv

header, *lines = open(input_path).readlines()
fout = open(output_path, 'w')

fout.write(header)

epsilon = 0.1
prev_offset = None
prev_text = None

for line in lines:
    offset, mode, text = json.loads(line)
    if prev_offset is None:
        prev_offset = offset
        prev_text = text
    elif offset - prev_offset < epsilon:
        prev_text += text
    else:
        fout.write(json.dumps([prev_offset, 'o', prev_text]) + '\n')
        prev_offset = offset
        prev_text = text

if prev_offset is not None:
    fout.write(json.dumps([prev_offset, 'o', prev_text]) + '\n')

