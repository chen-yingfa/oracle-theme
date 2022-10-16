# Data

Data preprocessing pipeline:

1. `preprocess.py`: Copy the `modern_text` data from meta data file on every shell, dump as JSON to `preprocessed/{data_name}`

2. `split_data.py`: Split into train, dev and test in 90:5:5 ratio, dump into three JSONs.
