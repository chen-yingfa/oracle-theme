import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class OracleNerDataset(Dataset):
    def __init__(
        self,
        examples_file: Path,
        tokenizer,
        labels_file: Path = Path("./ner/labels.json"),
        max_length: int = 512,
    ):
        self.labels_file = labels_file
        self.examples_file = examples_file
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_list = json.load(open(labels_file))
        self.bio_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.examples = self.get_examples(self.examples_file)
        self.features = self.get_features(self.examples)

    def get_examples(self, file: Path) -> List[dict]:
        """
        Load examples from file, return list of dict, where each dict is an
        example.
        """
        examples = []
        with open(file) as f:
            for line in f:
                # 往下走到第一个空行
                text = ""
                labels = []
                for line in f:
                    if line.strip() == "":  # 遇到空行
                        break
                    # NOTE: glyph might be a space " "
                    glyph, bio_label = line.rstrip().split("\t")
                    text += glyph
                    label = self.bio_to_id[bio_label]
                    labels.append(label)
                examples.append(
                    {
                        "text": text,
                        "labels": labels,
                    }
                )
        return examples

    def get_features(self, examples: List[dict]) -> dict:
        """
        Convert example text and labels in ID using tokenizer and a given
        list of labels.

        labels: (N, seq_len)
        input_ids: (N, seq_len)
        """
        all_texts = [ex["text"] for ex in examples]
        num_examples = len(examples)
        print(f"Tokenizing {num_examples} examples...")
        inputs = self.tokenizer(
            all_texts,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        # Build label tensor
        all_labels = torch.zeros(
            num_examples, inputs['input_ids'].shape[1], dtype=torch.long)
        for ex_idx, ex in enumerate(examples):
            for label_idx, label_id in enumerate(ex["labels"]):
                all_labels[ex_idx][label_idx] = label_id

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": all_labels,
        }

    def __getitem__(self, idx: int):
        return {key: self.features[key][idx] for key in self.features}

    def __len__(self) -> int:
        return len(self.features["input_ids"])
