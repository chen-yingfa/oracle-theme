import json

import torch
from torch.utils.data import Dataset

from tokenization.tokenizer import OracleThemeTokenizer


class OracleThemeDataset(Dataset):
    def __init__(self, file: str, tokenizer: OracleThemeTokenizer, max_length: int=512):
        self.file = file
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.examples = self.get_examples(file)
        self.features = self.get_features(self.examples, tokenizer)
        
    def get_examples(self, file) -> list:
        examples = []
        for line in open(file):
            data = json.loads(line)
            labels = data['theme'].split('/')
            labels = [t.strip() for t in labels]
            labels = [t for t in labels if len(t) > 0]
            example = {
                'text': data['oracle_text'],
                'labels': labels,
            }
            examples.append(example)
        return examples
    
    def get_label_list(self) -> list:
        return [
            '祭祀', '災難', '奴隸主貴族', '時間', 
            '方域', '氣象', '卜法', '漁獵、畜牧', '戰爭', 
            '鬼神崇拜', '商業、交通', '吉凶、夢幻', 
            '軍隊、刑法、監獄', '貢納', '農業', '官吏', 
            '飲食', '音樂', '死喪', '生育', 
            '疾病', '建築', '天文、曆法', 
            '奴隸、平民'
        ]
        
    def get_label2id(self) -> dict:
        label_list = self.get_label_list()
        return {label: i for i, label in enumerate(label_list)}
    
    def get_features(self, examples: list, tokenizer: OracleThemeTokenizer) -> dict:
        label2id = self.get_label2id()
        all_texts = [x['text'] for x in examples]
        
        # Build label one hot representation
        num_examples = len(examples)
        num_labels = len(label2id)
        all_labels = torch.zeros(num_examples, num_labels)
        for ex_idx, ex in enumerate(examples):
            # print(x)
            for label in ex['labels']:
                label_id = label2id[label]
                all_labels[ex_idx][label_id] = 1.0
        
        inputs = tokenizer(all_texts, self.max_length)
        features = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': all_labels,
        }
        return features
    
    def __getitem__(self, idx: int):
        return {
            key: self.features[key][idx] for key in self.features
        }
        
    def __len__(self) -> int:
        return len(self.features['input_ids'])

if __name__ == '__main__':
    file = 'data/dev.json'
    tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')
    data = OracleThemeDataset(file, tokenizer, 24)
    print(data[3])
    
