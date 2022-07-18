import torch
from torch import nn, LongTensor
from torch.autograd import Variable
import torch.nn.functional as F
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
            model_path, num_labels=num_labels, problem_type="multi_label_classification")
        
        # Init custom embedding layer, and deprecate pretrained embeddings
        # self.model.embeddings = nn.Embedding(vocab_size, hidden_size)
        # self.model.config.problem_type = 'multi_label_classification'
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels,
        )


class LSTMClassifier(nn.Module):
    def __init__(
        self, 
        num_labels: int=24,
        batch_size: int=32,
        embed_dim: int=768,
        hidden_dim: int=768,
        vocab_size: int=21000,
        ):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.hidden = self.init_hidden()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def init_hidden(self):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        return (h0, c0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        self.hidden = self.init_hidden()
        x = self.embeddings(input_ids)
        x = x.view(input_ids.shape[1], self.batch_size, -1)     # (L, B, H)
        # x = embeds.view(len(input_ids), self.batch_size, -1)
        # print(x.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        logits = self.hidden2label(lstm_out[-1])
        # print(logits.shape)
        # print(lstm_out.shape)
        # exit()
        loss = self.loss_fn(logits, labels)
        if labels is None:
            return {
                'logits': logits,
            }
        else:
            return {
                'logits': logits,
                'loss': loss,
            }


if __name__ == '__main__':
    model_path = 'hfl/rbt3'
    model = BertOracleTheme(model_path)
    # print(model)
    input_ids = LongTensor([
        [    1,     5,  2122,  5550,  1635,  8071, 11807, 15593,  1444,     5,
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
    