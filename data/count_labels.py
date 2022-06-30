from pathlib import Path
import json

def load_jsonl(filename) -> list:
    return [json.loads(line) for line in open(filename, 'r', encoding='utf-8')]

def load_label_map() -> dict:
    clusters = json.load(open('labels/label_clusters.json', encoding='utf-8'))
    map = {}
    for name, labels in clusters.items(): 
        for label in labels:
            map[label] = name
    return map

# Load data
data_dir = Path('./data')
# files = ['train.json', 'dev.json', 'test.json']
# examples = sum([load_jsonl(data_dir / f) for f in files], [])
files = ['ancient_0.json', 'ancient_1.json']
examples = sum([json.load(open(data_dir / f, encoding='utf8')) for f in files], [])
themes = [x['theme'] for x in examples]
label_map = load_label_map()

# Get all labels
labels = {}  # name -> occurrence
for i, theme in enumerate(themes):
    theme = [t.replace(' ', '') for t in theme.split('/')]
    for t in theme:
        t = t.strip()
        if t in label_map:
            t = label_map[t]
        else:
            t = ''
        if t not in labels:
            labels[t] = 0
        labels[t] += 1

label_list = list(labels.items())
label_list = sorted(label_list, key=lambda x: x[1], reverse=True)
print(label_list)
print(len(label_list))

with open('main_labels.txt', 'w', encoding='utf-8') as f:
    for name, occ in label_list:
        f.write(f'{name}\t{occ}\n')
    

labels = [x[0] for x in label_list]
print(labels)
