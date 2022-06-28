from pathlib import Path
import json


from model import BertOracleTheme
from tokenization.tokenizer import OracleThemeTokenizer
from dataset import OracleThemeDataset
from trainer import Trainer

output_dir = Path('result/temp')
output_dir.mkdir(exist_ok=True, parents=True)
data_dir = Path('data')


# Tokenizer
tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')

train_data = OracleThemeDataset(data_dir / 'train.json', tokenizer)
dev_data = OracleThemeDataset(data_dir / 'dev.json', tokenizer)
num_labels = len(train_data.get_label_list())

# Model
model = BertOracleTheme('hfl/rbt3', num_labels=num_labels)

# Train
trainer = Trainer(
    model=model.cuda(),
    output_dir=output_dir,
    batch_size=32,
    log_interval=20,
)
trainer.train(train_data, dev_data, resume=False)

# Test
test_data = OracleThemeDataset(data_dir / 'test.json', tokenizer)

