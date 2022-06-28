from pathlib import Path
import json
import random


def load(file: str) -> list:
    data = []
    with open(file) as f:
        next(f)
        for line in f:
            line = line.strip()[:-1]
            data.append(json.loads(line))
    return data


def dump(data: list, file: Path):
    with open(file, 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


data_dir = Path('data')
src_file = data_dir / 'ancient.json'

data = load(src_file)

# Shuffle and split data
random.seed(0)
random.shuffle(data)
cnt = len(data)
splits = [int(cnt * 0.9), int(cnt * 0.95)]
print(f'Num of examples: {cnt}')
print(f'Splits: {splits}')

train_data = data[:splits[0]]
dev_data = data[splits[0]:splits[1]]
test_data = data[splits[1]:]

print(f'train: {len(train_data)}')
print(f'dev: {len(dev_data)}')
print(f'test: {len(test_data)}')

print(f'Dumping to directory: {data_dir}')
dump(train_data, data_dir / 'train.json')
dump(dev_data, data_dir / 'dev.json')
dump(test_data, data_dir / 'test.json')
