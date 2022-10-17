import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import numpy as np


def get_metrics_1d(labels, preds):
    pos_label = 0
    pos_pred = 0
    pos_match = 0
    n = labels.shape[0]
    for i in range(n):
        if labels[i]:
            pos_label += 1
        if preds[i]:
            pos_pred += 1
            if labels[i]:
                pos_match += 1
    recall = pos_match / (pos_label + 1e-9)
    prec = pos_match / (pos_pred + 1e-9)
    f1 = 2 * recall * prec / (recall + prec + 1e-9)
    score = {
        "f1": f1,
        "prec": prec,
        "recall": recall,
    }
    return score


def get_metrics(labels, preds):
    n, m = labels.shape
    scores = []
    for j in range(m):
        score = get_metrics_1d(labels[:, j], preds[:, j])
        scores.append(score)
    return scores


# model_path = 'hfl/rbt3'
# model_path = 'lstm'
model_path = "jiagu_text_bert"
exp_name = "220629_handa"
# run_name = 'lr0.5_bs64_embed768_h512_ep4'
run_name = "lr5e-05"
result_dir = Path("result", exp_name, model_path, run_name)

test_dir = result_dir / "test"
preds_file = test_dir / "preds.json"
labels_file = test_dir / "labels.json"

preds = json.load(open(preds_file))
labels = json.load(open(labels_file))

# preds = np.array(preds)
preds = np.ones_like(labels)
labels = np.array(labels)

scores = get_metrics(labels, preds)

# labels = labels.flatten()
# preds = preds.flatten()
# scores = [get_metrics_1d(labels, preds)]
# print(f1_score(labels, preds))

print(scores)
f1s = [s["f1"] for s in scores]
print(f"Average F1: {round(100 * sum(f1s) / len(f1s), 2)}")
print(f1s)
plt.plot(f1s)
plt.xlabel("Theme")
plt.ylabel("F1")
plt.savefig("f1_theme.png")
plt.clf()
