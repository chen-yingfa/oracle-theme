import json
from pathlib import Path

import numpy as np


result_dir = Path('hfl/rbt3')
test_dir = result_dir / 'test'
preds_file = test_dir / 'preds.json'
labels_file = test_dir / 'labels.json'

preds = json.load(open(preds_file))
labels = json.load(open(labels_file))

preds = np.array(preds)
labels = np.array(labels)
print(preds.shape)
