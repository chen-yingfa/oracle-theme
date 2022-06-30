from pathlib import Path
import json


from model import BertOracleTheme
from tokenization.tokenizer import OracleThemeTokenizer
from dataset import OracleThemeDataset
from trainer import Trainer

# model_path = 'hfl/chinese-macbert-base'
model_path = 'hfl/rbt3'
exp_name = '220629'
output_dir = Path('result', model_path, exp_name)
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path('data/220629')


# Tokenizer
tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')

train_data = OracleThemeDataset(data_dir / 'train.json', tokenizer)
dev_data = OracleThemeDataset(data_dir / 'dev.json', tokenizer)
num_labels = len(train_data.get_label_list())

# Model
model = BertOracleTheme(model_path, num_labels=num_labels)

# Train
trainer = Trainer(
    model=model.cuda(),
    output_dir=output_dir,
    batch_size=32,
    grad_acc_steps=1,
    log_interval=20,
    num_epochs=12,
)
trainer.train(train_data, dev_data)

# Test
test_dir = output_dir / 'test'
test_dir.mkdir(exist_ok=True, parents=True)
test_data = OracleThemeDataset(data_dir / 'test.json', tokenizer)
test_result = trainer.evaluate(test_data, test_dir, 'test')
del test_result['preds']
print(test_result, flush=True)
json.dump(test_result, open(test_dir / 'result.json', 'w'))

