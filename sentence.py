from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from hazm import word_tokenize

class Sentence:
    def __init__(self, text, normalize=True) -> None:
        self._text:str = text        
        if normalize:
            self._text = self._normalize()
        

    def print(self):
        print(self._text)

    def _normalize(self) -> str:
        normalizer = BasicTextNormalizer()
        return normalizer(self._text)
    
    def _word_tokenize(self) -> list[str]:
        return list(word_tokenize(self._text))        

    def get(self, N) -> list[str]:
        result: list[str] = []
        for _ in range(N - 1):
            result.append("<s>")

        result.extend(self._word_tokenize())

        result.append("</s>")
            
        return result
    
        
