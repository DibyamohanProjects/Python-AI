from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token_to_id = {"<PAD>": 0}
        self.id_to_token = {0: "<PAD>"}

    def build(self, dataset):
        counter = Counter()

        for sample in dataset:
            counter.update(sample["tokens"])

        for token, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, tokens):
        return [
            self.token_to_id.get(token, 0)
            for token in tokens
        ]

    def __len__(self):
        return len(self.token_to_id)
