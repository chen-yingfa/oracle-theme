from torch import LongTensor
import torch


class OracleThemeTokenizer:
    '''A tokenizer that split each character into individual tokens.'''
    def __init__(self, vocab_file='./vocab.txt'):
        self.vocab = self.get_vocab(vocab_file)
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'  # Useless because we don't do MLM
        # print(list(self.vocab.keys())[:100])
    
    def get_vocab(self, file: str) -> dict:
        vocab = {}
        for idx, line in enumerate(open(file)):
            line = line.rstrip().split('\t')
            vocab[line[0]] = idx
        return vocab
    
    def tokenize(self, text: str) -> list:
        '''
        Turn into tokens
        '''
        tokens = list(text)
        for i, token in enumerate(tokens):
            if token not in self.vocab:
                tokens[i] = '[UNK]'
        return tokens
    
    def token_to_id(self, token):
        if token in self.vocab:
            return self.vocab[token]
        return '[UNK]'
    
    def encode(
        self, 
        text: str, 
        max_length: int=512) -> dict:
        '''
        Encode using BERT's method: 
            
            [CLS] + tokens + [SEP]
        
        but only one input sequence, and pad to max_length.
        '''
        tokens = self.tokenize(text)
        trunc_len = max_length - 2  # Account for CLS and SEP token
        tokens = tokens[:trunc_len]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = [self.token_to_id(token) for token in tokens]
        
        actual_len = len(token_ids)
        pad_len = max_length - actual_len
        att_mask = [1] * actual_len + [0] * pad_len
        token_ids += [0] * pad_len
        return {
            'input_ids': LongTensor(token_ids),
            'attention_mask': LongTensor(att_mask),
        }
    
    def encode_many(
        self,
        texts: list,
        max_length: int=512) -> dict:
        '''Encode multiple texts'''
        result = {
            'input_ids': [],
            'attention_mask': [],
        }
        for text in texts:
            one_result = self.encode(text, max_length)
            for key in result:
                result[key].append(one_result[key])
        cnt = len(result['input_ids'])
        for key in result:
            result[key] = torch.cat(result[key]).view((cnt, -1))
        return result
    
    def __call__(
        self, 
        input: str, 
        max_length: int=512,
        ) -> dict:
        '''
        Tokenize and encode into token IDs, and get attention masks.
        '''
        if isinstance(input, str):
            return self.encode(input, max_length)
        else:
            return self.encode_many(input, max_length)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    texts = [
        " 輣束鏛叵饭鏚掻 ",
        " 疄萍幗… ",
        " ",
        "",
        " …[胔]獢嗇槌伊峮倪椛欙刓听銺笵镈垷 ",
    ]
    print('Tokenization result')
    for text in texts:
        print(text)
        print(tokenizer.tokenize(text))
    print('Encoding result')
    result = tokenizer(texts, 12)
    for k, v in result.items():
        print(k)
        print(v)
    