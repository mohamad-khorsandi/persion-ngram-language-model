import re
from corpus import Corpus
from nltk.probability import LaplaceProbDist, SimpleGoodTuringProbDist
from nltk import FreqDist
import random
from sentence import Sentence

class LM():
    def __init__(self, train_corpus: Corpus, gram_count: int) -> None:
        self._N: int = gram_count
        self._train_corpus: Corpus = train_corpus

        self._ngrams:FreqDist = self.cal_n_grams(self._train_corpus, self._N)
        bins:int = len(self._train_corpus.get_vocab()) ** self._N
        self._laplace_dist:LaplaceProbDist = LaplaceProbDist(self._ngrams, bins=1e6)
        self._goodtur_dist:SimpleGoodTuringProbDist = SimpleGoodTuringProbDist(self._ngrams, bins=1e10)
    
    @classmethod
    def cal_n_grams(cls, cps: Corpus, n: int) -> FreqDist:
        result:FreqDist = FreqDist()

        for i in range(cps.get_len()):
            sen: list[str] = cps.get_item(i)
            for j in range(len(sen)-n+1):
                seq: list[str] = []
                
                for k in range(j, j+n):
                    seq.append(sen[k])

                key:str = ' '.join(seq)

                if key in result:
                    result[key] += 1
                else: 
                    result[key] = 1

        return result
    
    def cal_prob(self, ngram: str) -> float:
        assert len(ngram.split(' ')) == self._N
        if self._N == 1:
            return self._laplace_dist.prob(ngram)
        else:
            return self._goodtur_dist.prob(ngram)
    
    def perplexity(self, senObj:Sentence)-> float:
        sen: list[str] = senObj.get(self._N)
        prod = 1.0
        for i in range(len(sen) - self._N + 1):
            cur_g:tuple[str,...] = tuple(sen[i:i+self._N])
            p: float = self.cal_prob(' '.join(cur_g))
            prod *= p

        return (1/prod) ** (1/(len(sen) - self._N + 1))
        
    def get_n(self) -> int:
        return self._N
    
    def generate(self, prefix:list[str]) -> str:
        assert len(prefix) == self._N - 1 

        next_cadidates: list[str] = []
        next_probs:list[float] = []

        for word in self._train_corpus.get_vocab():
            prefix.append(word)
            next_cadidates.append(word)
            next_probs.append(self.cal_prob(' '.join(prefix)))
            prefix.pop()

        return random.choices(next_cadidates, weights=next_probs)[0]

        
    def extened_sen(self, begin:str) -> Sentence:
        new_sen: list[str] = begin.split(' ')
        while(len(new_sen) < 12):
            if self._N == 1:
                new_sen.append(self.generate([]))
            else:
                new_sen.append(self.generate(new_sen[-self._N+1:]))
        return Sentence(' '.join(new_sen), normalize=False)

