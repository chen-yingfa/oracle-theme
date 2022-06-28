from pathlib import Path
import json

from utils import load_jsonl


# Load data
data_dir = Path('data')
files = ['train.json', 'dev.json', 'test.json']
examples = sum([load_jsonl(data_dir / f) for f in files], [])
themes = [x['theme'] for x in examples]

# Get all labels
labels = {}  # name -> occurrence
for theme in themes:
    theme = [t.replace(' ', '') for t in theme.split('/')]
    for t in theme:
        t = t.strip()
        if t not in labels:
            labels[t] = 0
        labels[t] += 1

label_list = list(labels.items())
label_list = sorted(label_list, key=lambda x: x[1], reverse=True)
print(label_list)
print(len(label_list))

labels = [x[0] for x in label_list]
print(labels)
