from pathlib import Path
import json

from transformers import BertTokenizer

from model import BertOracleTheme, LSTMClassifier
# from tokenization.tokenizer import OracleThemeTokenizer
from dataset import OracleThemeDataset
from trainer import Trainer

# Hyperparams
lr = 0.5
batch_size = 64
hidden_dim = 512
embed_dim = 512
num_epochs = 4

# model_path = 'hfl/chinese-macbert-base'
model_path = 'lstm'
data_name = '220629_handa'
exp_name = f'lr{lr}_bs{batch_size}_embed{embed_dim}_h{hidden_dim}_ep{num_epochs}'
output_dir = Path('result', data_name, model_path, exp_name)
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path('data/preprocessed/220629')


# Tokenizer
# tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')
tokenizer = BertTokenizer.from_pretrained('hfl/rbt3')

train_data = OracleThemeDataset(data_dir / 'train.json', tokenizer)
dev_data = OracleThemeDataset(data_dir / 'dev.json', tokenizer)
num_labels = len(train_data.get_label_list())

# Model
# model = BertOracleTheme(model_path, num_labels=num_labels)
model = LSTMClassifier(
    num_labels=num_labels,
    batch_size=batch_size,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
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
# trainer.train(train_data, dev_data)

# Test
test_dir = output_dir / 'test'
test_dir.mkdir(exist_ok=True, parents=True)
test_data = OracleThemeDataset(data_dir / 'test.json', tokenizer)
trainer.load_best_ckpt()
test_result = trainer.evaluate(test_data, test_dir, 'test')
del test_result['preds']
print(test_result, flush=True)
json.dump(test_result, open(test_dir / 'result.json', 'w'))

