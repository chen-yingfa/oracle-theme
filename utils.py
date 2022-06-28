import json

def load_jsonl(file) -> list:
    return [json.loads(line) for line in open(file, 'r')]

def dump_jsonl(data: list, file):
    with open(file, 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

