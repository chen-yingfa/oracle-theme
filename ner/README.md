# NER using BiLSTM

## Code Structure

- `data.py`: Dataset 类，用于载入并处理 CSV 数据。
- `model.py`: 用于 NER 的 LSTM 模型，直接在 LSTM 上面加一个 token classifier 层。