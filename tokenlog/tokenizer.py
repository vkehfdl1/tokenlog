from abc import abstractmethod, ABC
from typing import List

import tiktoken
from transformers import AutoTokenizer


class BaseTokenizer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()

    @abstractmethod
    def _load_tokenizer(self):
        pass

    @abstractmethod
    def count_tokens(self, text: str):
        pass

    @abstractmethod
    def count_tokens_batch(self, texts: List[str]):
        pass


class TiktokenTokenizer(BaseTokenizer):
    def _load_tokenizer(self):
        return tiktoken.encoding_for_model(self.model_name)

    def count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        token_lists = self.tokenizer.encode_batch(texts, allowed_special='all')
        return list(map(len, token_lists))


class HuggingfaceTokenizer(BaseTokenizer):

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def count_tokens(self, text: str):
        tokens = self.tokenizer([text]).data['input_ids'][0]
        return len(tokens)

    def count_tokens_batch(self, texts: List[str]):
        token_lists = self.tokenizer(texts).data['input_ids']
        return list(map(len, token_lists))
