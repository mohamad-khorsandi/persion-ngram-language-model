from unittest import result
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from hazm import sent_tokenize, word_tokenize
from hazm import Normalizer

class Corpus:
    def __init__(self, corp_file: str, gram_count: int) -> None:
        assert gram_count <= 3 #send 0 for not adding any start or end symbol
        
        self._sen_corp: list[str] = []
        self._word_corp: list[list[str]] = []
        self._gram_count: int = gram_count
        self._vocab:set[str] = set()
        self.hazm = Normalizer()

        for comment in open(corp_file, encoding='utf-8').read().split('"'):
            for sen in sent_tokenize(comment):
                self._sen_corp.append(sen)
        
        self._sen_corp = self._sen_corp[1:]
        self._sen_corp = self._normalize()
        self._word_corp = self._word_tokenize()
        if (gram_count != 0):
            self._word_corp = self._add_special_toks()

    def print(self, word:bool=True) -> None:
        if word:
            for sen_t in self._word_corp:
                print(sen_t)
                print("--------")
        else:
            for sen in self._sen_corp:
                print(sen)
                print("--------")

    def _normalize(self) -> list[str]:
        normalizer = BasicTextNormalizer()
        new_c: list[str] = []
        
        for sen in self._sen_corp:
            new_c.append(self.hazm.remove_diacritics(normalizer(sen)))

        return new_c
    
    def _add_special_toks(self) -> list[list[str]]:
        result: list[list[str]] = []
        
        for sen in self._word_corp:
            new_sen: list[str] = []

            for _ in range(self._gram_count - 1):
                new_sen.append("<s>")
            new_sen.extend(sen)
            new_sen.append("</s>")

            result.append(new_sen)
        return result
    
    def _word_tokenize(self) -> list[list[str]]:
        result: list[list[str]] = []

        for sen in self._sen_corp:
            result.append(list(word_tokenize(sen)))
        return result
            
    def get_item(self, ind: int) -> list[str]:
        return self._word_corp[ind]
    
    def get_len(self) -> int:
        return len(self._sen_corp)
    
    def get_vocab(self) -> set[str]:
        if self._vocab:
            return self._vocab
        
        for sen in self._word_corp:
            for word in sen:
                self._vocab.add(word)
        return self._vocab

if __name__ == "__main__":
    c = Corpus("digikala_comment.csv", 1)
    c.print()
