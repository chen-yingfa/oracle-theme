from pathlib import Path
import json
import random

# NOTE: Change this
data_dir = Path('220629')
src_file = data_dir / 'ancient_220629.json'

data = json.load(open(src_file, encoding='utf8'))

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
json.dump(
    train_data,
    open(data_dir / 'train.json', 'w', encoding='utf8'), 
    ensure_ascii=False, indent=2)
json.dump(
    dev_data,
    open(data_dir / 'dev.json', 'w', encoding='utf8'), 
    ensure_ascii=False, indent=2)
json.dump(
    test_data,
    open(data_dir / 'test.json', 'w', encoding='utf8'), 
    ensure_ascii=False, indent=2)
