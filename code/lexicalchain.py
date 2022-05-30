import nltk
import string
from heapq import nlargest
from nltk.tag import pos_tag
from string import punctuation
from inspect import getsourcefile
from collections import defaultdict
from nltk.tokenize import word_tokenize
from os.path import abspath, join, dirname
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer

class Summarizer:
    
    def __init__(self, threshold_min=0.1, threshold_max=0.9):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max 
        self._stopwords = set(stopwords.words('english') + list(punctuation))
        

    def return_frequencies(self, words, lexical_chain):
        frequencies = defaultdict(int)
        for word in words:
            for w in word:
                if w not in self._stopwords:
                    flag = 0
                    for i in lexical_chain:
                        if w in list(i.keys()):
                            frequencies[w] = sum(list(i.values()))
                            flag = 1
                            break
                    if flag == 0: 
                        frequencies[w] += 1
        m = float(max(frequencies.values()))
        for w in list(frequencies.keys()):
            frequencies[w] = frequencies[w]/m
            if frequencies[w] >= self.threshold_max or frequencies[w] <= self.threshold_min:
                del frequencies[w]
        return frequencies

    def summarize(self, tokens, lexical_chain, n):
        idx = []
        assert n <= len(tokens)
        word_sentence = [word_tokenize(s.lower()) for s in tokens]
        self.frequencies = self.return_frequencies(word_sentence, lexical_chain)
        ranking = defaultdict(int)
        for i, sent in enumerate(word_sentence):
            for word in sent:
                if word in self.frequencies:
                    ranking[i] += self.frequencies[word]
                    idx = self.rank(ranking, n) 
        final_index = sorted(idx)
        return [tokens[j] for j in final_index]

    def rank(self, ranking, n):
        return nlargest(n, ranking, key=ranking.get)