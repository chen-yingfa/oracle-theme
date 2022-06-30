import torch
from torch import nn, LongTensor
from transformers import BertForSequenceClassification


class BertOracleTheme(nn.Module):
    def __init__(
        self, 
        model_path, 
        num_labels: int,
        vocab_size: int=21000, 
        hidden_size: int=768, 
        ):
        '''
        Multi-label classifier, pretrained encoder and pooler, randomly 
        initialized classifier and embedding.
        '''
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels)
        
        # Init custom embedding layer, and deprecate pretrained embeddings
        self.model.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.model.config.problem_type = 'multi_label_classification'
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels,
        )


if __name__ == '__main__':
    model_path = 'hfl/rbt3'
    model = BertOracleTheme(model_path)
    # print(model)
    input_ids = LongTensor([[    1,     5,  2122,  5550,  1635,  8071, 11807, 15593,  1444,     5,
             2,     0],
        [    1,     5, 19531, 10477,  9739,     6,     5,     2,     0,     0,
             0,     0],
        [    1,     5,     2,     0,     0,     0,     0,     0,     0,     0,
             0,     0],
        [    1,     2,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0],
        [    1,     5,     6,     8,  4863,     7,   184, 15886,  2170,  5867,
          2323,     2]])
    attention_mask = LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    outputs = model(**inputs)
    print(outputs)
    