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
            '災咎', '祭祀', '商王', '祭牲', '卜旬', '氣象', '兆語', '今時', 
            '卜夕', '殘辭', '地名', '田獵', '先王', '來時', '旬夕', '先妣',
            '月時', '使令', '往來', '捕獲', '祖先', '人名', '卜日', '數字', 
            '諸父', '方國', '祈年', '先公', '告呼', '禱祝', '禳禦', '征伐', 
            '商臣', '疾夢', '前往', '聯合', '方名', '水神', '諸婦', '出步', 
            '日時', '祭品', '軍事', '遘遇', '祰祭', '人牲', '族衆', '延續', 
            '出入', '動物', '方位', '問捷', '求雨', '牲畜', '走獸', '冊告',
            '省視', '悔憂', '宗廟', '死喪', '禳御', '諸示', '驗視', '諸母', 
            '侯伯', '諸兄', '農事', '取予', '出遊', '貞若', '諸子', '時間', 
            '諸祖', '往步', '天帝', '協事', '山神', '驗辭', '禮樂', '徵集', 
            '往出', '神祗', '工事', '追捕', '占馬', '宴饗', '時名', '寧災', 
            '傷亡', '貞祐', '舊臣', '奏告', '習語', '邑鄙', '生育', '獲得', 
            '心理', '執獲', '周祭', '處所', '玉器', '涉水', '漁獵', '器具', 
            '戍士', '史官', '鬼神', '俘虜', '蝗災', '開拔', '社神', '神祇', 
            '犬官', '宅居', '貢納', '病愈', '臣庶', '祭所', '獻享', '館舍', 
            '諸臣', '建築', '出巡', '侵掠', '田耕', '文字', '安撫', '敗降', 
            '山麓', '軍隊', '亞官', '尹官', '水文', '季時', '休息', '天文', 
            '射士', '頒賜', '教學', '神使', '方族', '逃逸', '種植', '職官', 
            '來往', '天象', '戒備', '貞事', '奴隸', '舉冊', '擇取', '尸主', 
            '工民', '方土', '師官', '樂官', '敵方', '水名', '領途', '出犯', 
            '駐次', '會同', '諸妣', '田官', '先祖', '畜牧', '弓矢', '受祐', 
            '開墾', '卜吉', '祈保', '方国', '奉將', '諸官', '蝗蟲', '巫師', 
            '植物', '保官', '氏族', '刑罰', '飛禽', '騎兵', '匹配', '駐舍', 
            '攜送', '方王', '輔佐', '神主', '截擊', '立中', '其他', '兵器', 
            '返歸', '保衛', '風名', '延長福祺', '使者', '作官', '淹死', '樂器', 
            '舉事', '離開', '祈雨', '逮住', '数字', '兵員', '簽名', '監獄', 
            '祰祭先公', '告乎', '延緩', '向', '抵禦', '真若', '還有', '咋就', 
            '祭祀字', '司官', '方神', '雲神', '投獻', '王事', '夕時', '災咎捕獲', 
            '处所', '神祗求雨', '質子', '諸婦祭祀', '四月', '迎候', '室名', '征集', 
            '子女', '月时', '祈求', '保神', '行为', '先王卜夕', '占事', '祭牲田獵', 
            '倉廩', '卜辭習語', '諸公', '卜月', '防守', '来时', '神位', '收穫',
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
    
