import torch
from torch import nn, LongTensor
from torch.autograd import Variable


class LstmForNer(nn.Module):
    def __init__(
        self,
        num_labels: int = 24,
        batch_size: int = 32,
        embed_dim: int = 768,
        hidden_dim: int = 768,
        vocab_size: int = 21000,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.d = 2 if self.bidirectional else 1

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=bidirectional,
            dropout=0.1,
        )
        self.hidden2label = nn.Linear(self.d * hidden_dim, num_labels)
        self.hidden = self.init_hidden()
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()

    def init_hidden(self):
        h0 = Variable(
            torch.zeros(self.d, self.batch_size, self.hidden_dim).cuda()
        )
        c0 = Variable(
            torch.zeros(self.d, self.batch_size, self.hidden_dim).cuda()
        )
        return (h0, c0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        L: sequence length
        B: batch size
        H: hidden size
        C: number of classes
        """
        self.hidden = self.init_hidden()
        x = self.embeddings(input_ids)
        x = x.view(input_ids.shape[1], self.batch_size, -1)  # (L, B, H)
        lstm_out, self.hidden = self.lstm(x, self.hidden)  # (L, B, H)
        logits = self.hidden2label(lstm_out[-1])  # (L, B, C)
        loss = self.loss_fn(logits, labels)
        if labels is None:
            return {
                "logits": logits,
            }
        else:
            return {
                "logits": logits,
                "loss": loss,
            }


if __name__ == "__main__":
    # model_path = 'hfl/rbt3'
    # model = BertOracleTheme(model_path)
    model = LstmForNer(
        num_labels=25,
    )
    # print(model)
    input_ids = LongTensor(
        [
            [1, 5, 2122, 5550, 1635, 8071, 11807, 15593, 1444, 5, 2, 0],
            [1, 5, 19531, 10477, 9739, 6, 5, 2, 0, 0, 0, 0],
            [1, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 5, 6, 8, 4863, 7, 184, 15886, 2170, 5867, 2323, 2],
        ]
    )
    attention_mask = LongTensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    outputs = model(**inputs)
    print(outputs)
