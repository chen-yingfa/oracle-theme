import json


def load_json(file):
    return json.load(open(file))


def dump_json(data, file):
    json.dump(data, open(file, 'w', encoding='utf8'), ensure_ascii=False)


def load_jsonl(file) -> list:
    return [json.loads(line) for line in open(file, 'r', encoding='utf8')]


def dump_jsonl(data: list, file):
    with open(file, 'w', encoding='utf8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def get_param_count(model) -> int:
    cnt = 0
    for k, v in model.named_parameters():
        cnt += v.numel()
    return cnt