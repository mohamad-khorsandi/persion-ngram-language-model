from ast import main
from hmac import new
from operator import ne
from typing import List
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from hazm import sent_tokenize, word_tokenize

class Corpus:
    def __init__(self, corp_file: str, gram_count: int) -> None:
        self._sen_corp: list[str] = []
        self._word_corp: List[List[str]] = []
        self._gram_count: int = gram_count
    
        for comment in open(corp_file).read().split('"'):
            for sen in sent_tokenize(comment):
                self._sen_corp.append(sen)
        
        self._sen_corp = self._sen_corp[1:]
        self._sen_corp = self._normalize()
        self._word_corp = self._word_tokenize()
        self._word_corp = self._add_special_toks()
            
    def print(self) -> None:
        for sen in self._word_corp:
            print(sen)
            print("--------")

    def _normalize(self) -> list[str]:
        normalizer = BasicTextNormalizer()
        new_c: list[str] = []
        
        for sen in self._sen_corp:
            new_c.append(normalizer(sen))

        return new_c
    
    def _add_special_toks(self) -> List[List[str]]:
        result: list[list[str]] = []
        
        for sen in self._word_corp:
            new_sen: list[str] = []

            for _ in range(self._gram_count):
                new_sen.append("<s>")
            new_sen.extend(sen)
            new_sen.append("</s>")

            result.append(new_sen)
        return result
    
    def _word_tokenize(self) -> List[List[str]]:
        result: list[list[str]] = []

        for sen in self._sen_corp:
            result.append(list(reversed(word_tokenize(sen))))
        return result

    def get_item(self, idx: int) -> List[str]:
        return word_tokenize(self._sen_corp[idx])
    
    def get_len(self) -> int:
        return len(self._sen_corp)


if __name__ == "__main__":
    c = Corpus("digikala_comment.csv", 1)
    c.print()
