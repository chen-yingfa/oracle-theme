from pathlib import Path
import json
import argparse

from transformers import BertTokenizer

from ner.model import LstmForNer
from ner.data import OracleNerDataset
from trainer import Trainer


# Hyperparams
lr = 0.1
batch_size = 64
hidden_dim = 512
embed_dim = 512
num_epochs = 8

# model_path = 'hfl/chinese-macbert-base'
model_path = "bilstm"
bidirectional = model_path == "bilstm"
data_name = "ner"
exp_name = (
    f"lr{lr}_bs{batch_size}_embed{embed_dim}_h{hidden_dim}_ep{num_epochs}"
)
output_dir = Path("result", data_name, model_path, exp_name)
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path("./ner")


# Tokenizer
# tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')
tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")

train_data = OracleNerDataset(data_dir / "train.csv", tokenizer)
dev_data = OracleNerDataset(data_dir / "dev.csv", tokenizer)
dev_data = None
num_labels = len(train_data.label_list)

# Model
# model = BertOracleTheme(model_path, num_labels=num_labels)
model = LstmForNer(
    num_labels=num_labels,
    batch_size=batch_size,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    bidirectional=bidirectional,
)

# Train
trainer = Trainer(
    model=model.cuda(),
    output_dir=output_dir,
    batch_size=batch_size,
    grad_acc_steps=1,
    log_interval=100,
    num_epochs=num_epochs,
    lr=lr,
    data_drop_last=True,
)
trainer.train(train_data, dev_data)

# Test
test_dir = output_dir / "test"
test_dir.mkdir(exist_ok=True, parents=True)
test_data = OracleNerDataset(data_dir / "test.csv", tokenizer)
trainer.load_best_ckpt()
test_result = trainer.evaluate(test_data, test_dir, "test")
del test_result["preds"]
print(test_result, flush=True)
json.dump(test_result, open(test_dir / "result.json", "w"))
