from pathlib import Path
import json

from transformers import BertTokenizer

from model import BertOracleTheme, JiaguTextBert
from tokenization.tokenizer import OracleThemeTokenizer
from dataset import OracleThemeDataset
from trainer import Trainer

# Hyperparams
lr = 2e-4

# model_path = 'hfl/chinese-macbert-base'
model_path = '/data/private/chenyingfa/models/jiagu_text_bert'
# model_path = 'hfl/rbt6'
data_name = '220629_handa'
exp_name = f'lr{lr}'
output_dir = Path('result', data_name, model_path, exp_name)
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path('data/preprocessed/220629')


# Tokenizer
# tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')
tokenizer = BertTokenizer.from_pretrained(model_path)

# Data
train_data = OracleThemeDataset(data_dir / 'train.json', tokenizer)
dev_data = OracleThemeDataset(data_dir / 'dev.json', tokenizer)
num_labels = len(train_data.get_label_list())

# Model
# model = BertOracleTheme(model_path, num_labels=num_labels)
model = JiaguTextBert(model_path, num_labels=num_labels)


# Train
trainer = Trainer(
    model=model.cuda(),
    output_dir=output_dir,
    batch_size=16,
    grad_acc_steps=2,
    log_interval=100,
    num_epochs=8,
    lr=lr,
)
trainer.train(train_data, dev_data)

# Test
test_dir = output_dir / 'test'
test_dir.mkdir(exist_ok=True, parents=True)
test_data = OracleThemeDataset(data_dir / 'test.json', tokenizer)
trainer.load_best_ckpt()
test_result = trainer.evaluate(test_data, test_dir, 'test')
del test_result['preds']
print(test_result, flush=True)
json.dump(test_result, open(test_dir / 'result.json', 'w'))
