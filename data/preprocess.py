from pathlib import Path
import json


HANDA_META_DIR = Path('./handa_meta')
data_names = ['B', 'D', 'H', 'HD', 'L', 'S', 'T', 'W', 'Y']


def dump_jsonl(data, file):
    with open(file, 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def remove_html(text: str) -> str:
    ret = ''
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '<':
            j = i
            while j < n and text[j] != '>': j += 1
            i = j + 1
        else:
            ret += text[i]
            i += 1
    return ret


def load_book_to_text(data_name: str) -> dict:
    '''
    Load meta data of a dataset and get map from book_name to modern_text
    '''
    meta_file = HANDA_META_DIR / f'oracle_meta_{data_name}.json'
    meta_data = json.load(open(meta_file))
    book_to_meta = {}
    for entry in meta_data:
        book_name = entry['book_name']
        if book_name not in book_to_meta:
            book_to_meta[book_name] = []
        meta = {key: entry[key] for key in ['row_order', 'modern_text']}
        book_to_meta[book_name].append(meta)
    
    # Merge different rows (条)
    book_to_text = {}
    for book_name, book_meta in book_to_meta.items():
        book_meta = sorted(book_meta, key=lambda x: x['row_order'])
        modern_text = '；'.join([x['modern_text'] for x in book_meta])
        book_to_text[book_name] = remove_html(modern_text)
        
    # print(book_to_text)
    map_file = HANDA_META_DIR / f'book_to_text_{data_name}.json'
    json.dump(book_to_text, open(map_file, 'w'), ensure_ascii=False, indent=2)
    return book_to_text


def load_all_book_to_text() -> dict:
    book_to_text = {}
    for data_name in data_names:
        print(f'processing {data_name}')
        book_to_text.update(load_book_to_text(data_name))
    return book_to_text


def preprocess(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(exist_ok=True, parents=True)
    file_names = ['train.json', 'dev.json', 'test.json']
    book_to_text = load_all_book_to_text()
    json.dump(
        book_to_text,
        open(HANDA_META_DIR / 'book_to_text.json', 'w'), 
        indent=2, ensure_ascii=False)
    for fname in file_names:
        src_file = src_dir / fname
        dst_file = dst_dir / fname
        examples = json.load(open(src_file))
        processed = []
        for ex in examples:
            book = ex['oracle_no']
            if not book or book not in book_to_text: continue
            processed.append({
                'book_name': book,
                'text': book_to_text[book],
                'theme': ex['theme'],
                'oracle_text': ex['oracle_text'],
            })
        
        print(f'Dumping {len(processed)} examples to {dst_file}')
        # dump_jsonl(processed, dst_file)
        json.dump(processed, open(dst_file, 'w'), ensure_ascii=False, indent=2)
    


if __name__ == '__main__':
    data_name = '220629'
    src_dir = Path(data_name)
    dst_dir = Path('preprocessed', data_name)
    preprocess(src_dir, dst_dir)