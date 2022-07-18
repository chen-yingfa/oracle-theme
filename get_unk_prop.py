from transformers import BertTokenizer
from tokenization.tokenizer import OracleThemeTokenizer
import json

path = 'hfl/rbt3'
# tokenizer = BertTokenizer.from_pretrained(path)
tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')
data_path = 'data/preprocessed/220629/dev.json'
examples = json.load(open(data_path))
texts = [x['text'] for x in examples]
unk = '[UNK]'

token_cnt = 0
unk_cnt = 0

for text in texts:
    tokens = tokenizer.tokenize(text)
    token_cnt += len(tokens)
    unk_cnt += tokens.count(unk)

print(token_cnt, unk_cnt, unk_cnt / token_cnt)
