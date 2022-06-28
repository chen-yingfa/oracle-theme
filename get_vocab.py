'''
Loop through all text in training data and get the set of all characters
'''
from pathlib import Path
import json


def load_jsonl(file) -> list:
    return [json.loads(line) for line in open(file)]


def load_data(data_dir: Path) -> list:
    data = load_jsonl(data_dir / 'train.json') + load_jsonl(data_dir / 'dev.json')
    return data


# Load data
data_dir = Path('data')
examples = load_data(data_dir)

# Build vocab
vocab = {}  # char -> occurrence
print(len(examples))
for ex in examples:
    for c in ex['oracle_text']:
        if c not in vocab:
            vocab[c] = 0
        vocab[c] += 1
print(len(vocab))
vocab_list = [(k, v) for k, v in vocab.items()]
vocab_list = sorted(vocab_list, key=lambda x: x[1], reverse=True)
special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']
vocab_list = [(t, 0) for t in special_tokens] + vocab_list

print(vocab_list[:100])

# Dump vocab
vocab_file = Path('tokenization/vocab.txt')
with open(vocab_file, 'w') as f:
    for ch, occ in vocab_list:
        f.write(f'{ch}\t{occ}\n')

