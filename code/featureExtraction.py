import heapq
import os
from pydoc import doc
from tkinter.ttk import Separator
import numpy as np
import json
import os
import nltk
import math
import re
from genericpath import exists
from bs4 import BeautifulSoup
from inspect import getsourcefile
from nltk.corpus import wordnet
from collections import defaultdict
from lexicalchain import Summarizer

class FeatureSelection:
    def __init__(self):
        self.files_C = []
        self.files_NC = []
        self.docs_C = {}                      # docs = {'doc1': [t1, t2, t3], 'doc2': [tokens], ...}
        self.docs_NC = {}                      # docs = {'doc1': [tokens], 'doc2': [tokens], ...}
        self.index_Course = {}
        self.index_NonCourse = {}
        self.features100C = []
        self.features100NC = []
        self.featureVectors_C = {}
        self.featureVectors_NC = {}
        self.noun50_C = []
        self.noun50_NC = []
        self.neighbours50_C = []
        self.neighbours50_NC = []
        self.vocab_C = []
        self.vocab_NC = []
        self.vocabVectors_C = {}
        self.vocabVectors_NC = {}
        self.LC_Vectors_C = {}
        self.LC_Vectors_NC = {}
        self.dataset = []
        self.separated = {}
        self.punctuations = "[.,!?:;\')(}{@#!$%^&*|?></~`+-_=‘’”“\"]"
        self.lemmatizer = nltk.WordNetLemmatizer()                  # nltk lemmatizer
        self.stopwords = ['a', 'is', 'the', 'of', 'all', 'and', 'to', 'can', 'be', 'as', 'once', 'for', 'at', 'am', 'are', 'has', 'have', 'had', 'up', 'his', 'her', 'in', 'on', 'no', 'we', 'do']

    def getTextFromHTML(self, path):
        url = open(path)
        html = url.read()
        soup = BeautifulSoup(html, features="html.parser")
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def getTokensFromFiles(self, text):
        text = text.lower()                                               # reading doc in text as str
        text = re.sub(self.punctuations, " ", text)                       # removing punctuations
        tokens = nltk.word_tokenize(text)                                 # tokenizing text and converting to list
        tokens = [self.lemmatizer.lemmatize(each) for each in tokens]     # lemmatizing terms/tokens
        return (tokens)

    def createIndex(self):
        count = 1
        path = "/course-cotrain-data/"
        cwd = os.getcwd()           # /Assignment 3
        p = cwd + path              # p = /Assignment 3/course-cotrain-data/
        os.chdir(p)                 # current dir = /Assignment 3/course-cotrain-data/

        ccd = ['fulltext'] # os.listdir(os.getcwd())       # ccd = ['fulltext', 'inlinks']
        for c in ccd:                       # c1 = 'fulltext', c2 = 'inlinks' - c1, c2 are iterations
            contents = os.listdir(os.path.join(os.getcwd(), c))     # contents = ['course', 'non-course']
            for content in contents:        # content1 = 'course', content2 = 'non-course' - content1, content2 are iterations
                index = {}
                docs = {}
                features100 = {}
                noun50 = {}
                count = 0 
                files = os.listdir(os.path.join(os.getcwd(), c, content))          # "files" contains a list of html files from content
                if content == 'course':
                    self.files_C = files
                if content == 'non-course':
                    self.files_NC = files
                for f in files:                                                    # visit each file in "files"
                    # print(len(files))
                    # print(os.path.join(os.getcwd(), c, content, f))
                    text = self.getTextFromHTML(os.path.join(os.getcwd(), c, content, f))
                    tokens = self.getTokensFromFiles(text)                         # getting tokens from each document
                    # print("count = ", count, " file: ", f)
                    for word in tokens:                                            # loop to iterate all tokens of each document
                        if(word not in self.stopwords):                            # removing stopwords from tokens
                            if(word not in index):                                 # condition to add new word in index (datatype: dict)
                                index[word] = {'TFs': [0]*len(files), 'TF-Sum': 0, 'DF': 0, 'IDF': 0, 'TF-IDFs': [0]*len(files), 'TF-IDF-Sum': 0} # making format of the dictionary for every distinct term
                                index[word]['TFs'][count - 1] += 1                 # counting Term Frequency of each term
                                index[word]['DF'] += 1                             # counting Document Frequency of each term
                            else:
                                if(index[word]['TFs'][count - 1] >= 1):            # if term already in index and Term has occured previously in this document
                                    index[word]['TFs'][count - 1] += 1             # increase TF only
                                else:
                                    index[word]['TFs'][count - 1] += 1             # else if Term is occuring first time in this document
                                    index[word]['DF'] += 1                         # increase TF and DF both
                            if(f not in docs):
                                docs[f] = []
                            docs[f].append(word)

                    count += 1
                for i in index:
                    index[i]['IDF'] = math.log(index[i]['DF'], 10)/len(files)   # calculating IDF of every term using formula: IDF = log(N/DF)
                for i in index:
                    for r in range(len(files)):
                        index[i]['TF-IDFs'][r] = index[i]['TFs'][r] * index[i]['IDF']   # calculating TFxIDF of each term in dictionary/index
                        index[i]['TF-IDF-Sum'] += index[i]['TF-IDFs'][r]                # calculating sum of TFxIDF each term in dictionary/index
                        index[i]['TF-Sum'] += index[i]['TFs'][r]                        # calculating sum of TFs each term in dictonary/index
                tempFt= {}

                for i in index:
                    tempFt[i] = index[i]['TF-IDF-Sum']

                features100 = heapq.nlargest(100, tempFt.keys())
                featureVectors = {}
                for f in files:
                    featureVectors[f] = [0]*len(features100)
                    count = 0
                    for word in features100:
                        if word in docs[f]:
                            featureVectors[f][count] += 1
                        else:
                            pass
                        count += 1

                print(len(docs.items()))

                is_noun = lambda pos: pos[:2] == 'NN'
                terms = list(index.keys())
                nouns = [word for (word, pos) in nltk.pos_tag(terms) if is_noun(pos)]

                tempN = {}

                for word in nouns:
                    tempN[word] = index[word]['TF-Sum']

                noun50 = heapq.nlargest(50, tempN.keys())


                tempTS = {}

                for n in noun50:                                # docs = {'doc1': [token1, token2, token3, ...], 'doc2': [token1, token2, token3, ...], ...}
                    for doc in docs.values():                   # docs.values() = [[token1, token2, token3, ...], [token1, token2, token3, ...], ...}
                        for w in doc:                           # doc = [token1, token2, token3, ...], w = token1
                            if ((n == w) and (w not in tempTS)):
                                if (doc.index(w) >= 1):
                                    indexP = doc.index(w) - 1
                                    tempTS[doc[indexP]] = index[doc[indexP]]['TF-Sum']
                                if (doc.index(w) < len(doc) - 1):
                                    indexN = doc.index(w) + 1
                                    tempTS[doc[indexN]] = index[doc[indexN]]['TF-Sum']
                
                neighbours50 = heapq.nlargest(50, tempTS.keys())

                vocab = noun50 + neighbours50
                
                vocabVectors = {}
                for f in files:
                    vocabVectors[f] = [0]*len(vocab)
                    count = 0
                    for word in vocab:
                        if word in docs[f]:
                            vocabVectors[f][count] += 1
                        else:
                            pass
                        count += 1

                relation = relation_list(nouns)
                lexical = create_lexical_chain(nouns, relation)
                final_chain = prune(lexical)

                tempList = []
                for f in files:
                    text = self.getTextFromHTML(os.path.join(os.getcwd(), c, content, f))
                    tokens = self.getTokensFromFiles(text)
                    if len(tokens) >= len(vocab):
                        n = len(vocab)
                    else: 
                        n = 2
                    fs = Summarizer()
                    for s in fs.summarize(tokens, final_chain, n):
                        tempList.append(s)
                    
                LC_Vectors = {}
                for f in files:
                    LC_Vectors[f] = [0]*len(tempList)
                    count = 0
                    for word in tempList:
                        if word in docs[f]:
                            LC_Vectors[f][count] += 1
                        else:
                            pass
                        count += 1

                if (content == "course"):
                    self.index_Course = index
                    self.features100C = features100
                    self.featureVectors_C = featureVectors
                    self.docs_C = docs
                    self.noun50_C = noun50
                    self.neighbours50_C = neighbours50
                    self.vocab_C = vocab
                    self.vocabVectors_C = vocabVectors
                    self.LC_Vectors_C = LC_Vectors
                elif (content == "non-course"):
                    self.index_NonCourse = index
                    self.features100NC = features100
                    self.featureVectors_NC = featureVectors
                    self.docs_NC = docs
                    self.noun50_NC = noun50
                    self.neighbours50_NC = neighbours50
                    self.vocab_NC = vocab
                    self.vocabVectors_NC = vocabVectors
                    self.LC_Vectors_NC = LC_Vectors

        self.separated['C'] = []
        self.separated['NC'] = []
        for f in self.files_C:
            self.dataset.append(self.featureVectors_C[f])
            self.dataset.append(self.vocabVectors_C[f])
            self.dataset.append(self.LC_Vectors_C[f])
            self.separated['C'].append(self.featureVectors_C[f])
            self.separated['C'].append(self.vocabVectors_C[f])
            self.separated['C'].append(self.LC_Vectors_C[f])

        for f in self.files_NC:
            self.dataset.append(self.featureVectors_NC[f])
            self.dataset.append(self.vocabVectors_NC[f])
            self.dataset.append(self.LC_Vectors_NC[f])
            self.separated['NC'].append(self.featureVectors_NC[f])
            self.separated['NC'].append(self.vocabVectors_NC[f])
            self.separated['NC'].append(self.LC_Vectors_NC[f])
        print("separated: ", self.separated)

        os.chdir(os.path.join(os.getcwd(), ".."))                          # return to the "Assignment 3" directory 

def relation_list(nouns):

    relation_list = defaultdict(list)

    for k in range (len(nouns)):   
        relation = []
        for syn in wordnet.synsets(nouns[k], pos = wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list

def create_lexical_chain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):
                    if key == noun and flag == 0:
                        lexical[j][noun] +=1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0: 
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical
 
def prune(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1: 
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain



model = FeatureSelection()
# model.createIndex()
# model.loadORcreateINDEXES()