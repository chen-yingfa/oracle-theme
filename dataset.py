import json

import torch
from torch.utils.data import Dataset

from tokenization.tokenizer import OracleThemeTokenizer


def load_label_map(clusters_file: str) -> dict:
    clusters = json.load(open(clusters_file, encoding="utf-8"))
    map = {}
    for name, labels in clusters.items():
        for label in labels:
            map[label] = name
    return map


class OracleThemeDataset(Dataset):
    def __init__(
        self,
        file: str,
        tokenizer: OracleThemeTokenizer,
        max_length: int = 512,
        clusters_file: str = "data/labels/label_clusters.json",
    ):
        self.file = file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clusters_file = clusters_file

        self.theme_clusters = json.load(
            open(clusters_file, "r", encoding="utf8")
        )
        self.examples = self.get_examples(file, clusters_file)
        self.features = self.get_features(self.examples, tokenizer)

    def get_examples(
        self,
        file: str,
        clusters_file: str = None,
        input_key: str = "text",
    ) -> list:
        theme_map = load_label_map(clusters_file)
        examples = []
        lines = json.load(open(file, encoding="utf8"))
        # for line in open(file):
        for entry in lines:
            # entry = json.loads(line)
            themes = entry["theme"].split("/")
            labels = []
            for i, theme in enumerate(themes):
                theme = theme.strip()
                if theme == "":
                    continue
                if theme in theme_map:
                    labels.append(theme_map[theme])
            example = {
                "text": entry[input_key],
                "labels": labels,
            }
            examples.append(example)
        return examples

    def get_label_list(self) -> list:
        return self.theme_clusters.keys()

    def get_label2id(self) -> dict:
        label_list = self.get_label_list()
        return {label: i for i, label in enumerate(label_list)}

    def get_features(
        self, examples: list, tokenizer: OracleThemeTokenizer
    ) -> dict:
        label2id = self.get_label2id()
        all_texts = [x["text"] for x in examples]

        # Build label one hot representation
        num_examples = len(examples)
        num_labels = len(label2id)
        all_labels = torch.zeros(num_examples, num_labels)
        for ex_idx, ex in enumerate(examples):
            # print(x)
            for label in ex["labels"]:
                label_id = label2id[label]
                all_labels[ex_idx][label_id] = 1.0

        print(f"Tokenizing {len(all_texts)} texts...")
        # inputs = tokenizer(all_texts, self.max_length)
        inputs = tokenizer(
            all_texts,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        features = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": all_labels,
        }
        return features

    def __getitem__(self, idx: int):
        return {key: self.features[key][idx] for key in self.features}

    def __len__(self) -> int:
        return len(self.features["input_ids"])


if __name__ == "__main__":
    file = "data/preprocessed/220629/dev.json"
    # tokenizer = OracleThemeTokenizer('tokenization/vocab.txt')
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")
    data = OracleThemeDataset(file, tokenizer, max_length=24)
    print(data[3])
