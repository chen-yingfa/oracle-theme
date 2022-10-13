# Oracle Theme

对甲骨文进行主题分类。

模型：
- LSTM
- BERT-Chinese
- JiaguTextBERT（宋晨阳）

## Data


应该将 train、dev 和 test 数据放在同一个目录，比如：

- `./my/data/path/train.json`
- `./my/data/path/dev.json`
- `./my/data/path/test.json`

然后在 `train.py` 的第 20 行的 `data_dir` 设为数据所在路径：

```python
# ...
data_dir = Path('./my/data/path')
# ...
```


数据格式：

```json
[
  {
    "book_name": "H26018",
    "text": "鼎（貞）：翼（翌）…其冓（遘）…歲于■…",
    "theme": "來時 / 祭祀",
    "oracle_text": " 嗀瓠…磦昅…厺枂… "
  },
  {
    "book_name": "H27996",
    "text": " 庚…； 叀（惠）用沙于止（翦）方，不雉眾。； 戍※方戍。； 弗（翦）。； 戍…（翦）。",
    "theme": "問捷 / 方國",
    "oracle_text": " 域…譫笘 "
  },
  // ...
]
```

其中 `theme` 字段就是 label

> 目前无视 `oracle_text` 字段。


## Execution

### Training

执行：`python3 train.py`

### Testing

同上，将来会单独把 test 代码分离出来的。
