from pathlib import Path
import json

from utils import load_jsonl


# Load data
data_dir = Path("data/220629")
# files = ['train.json', 'dev.json', 'test.json']
# examples = sum([load_jsonl(data_dir / f) for f in files], [])
# files = ['ancient_0.json', 'ancient_1.json']
files = ["ancient_220629.json"]
examples = sum(
    [json.load(open(data_dir / f, encoding="utf8")) for f in files], []
)
themes = [x["theme"] for x in examples]

# Get all labels
labels = {}  # name -> occurrence
for i, theme in enumerate(themes):
    theme = [t.replace(" ", "") for t in theme.split("/")][:1]
    for t in theme:
        # if t == '行为':
        #     print(i)
        #     exit()
        t = t.strip()
        if t not in labels:
            labels[t] = 0
        labels[t] += 1

print(labels)

label_list = list(labels.items())
label_list = sorted(label_list, key=lambda x: x[1], reverse=True)
print(label_list)
print(len(label_list))

with open("main_labels.txt", "w", encoding="utf-8") as f:
    for name, occ in label_list:
        f.write(f"{name}\t{occ}\n")


labels = [x[0] for x in label_list]
print(labels)
