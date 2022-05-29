import heapq
import os
import numpy as np
import json
import os
import nltk
import math
import re
from genericpath import exists
from bs4 import BeautifulSoup

class FeatureSelection:
    def __init__(self):
        self.index_Course = {}
        self.index_NonCourse = {}
        self.features100C = {}
        self.features100NC = {}
        self.numOfDocs = (230 + 821)# * 2
        self.punctuations = "[.,!?:;‘’”“\"]"
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

    # def indexes(self, index, f, files, count, ):

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
                features100 = {}
                count = 0 
                files = os.listdir(os.path.join(os.getcwd(), c, content))          # "files" contains a list of html files from content
                for f in files:                                                    # visit each file in "files"
                    # print(len(files))
                    # print(os.path.join(os.getcwd(), c, content, f))
                    text = self.getTextFromHTML(os.path.join(os.getcwd(), c, content, f))
                    tokens = self.getTokensFromFiles(text)                         # getting tokens from each document
                    print("count = ", count, " file: ", f)
                    for word in tokens:                                            # loop to iterate all tokens of each document
                        if(word not in self.stopwords):                            # removing stopwords from tokens
                            if(word not in index):                                 # condition to add new word in index (datatype: dict)
                                index[word] = {'TFs': [0]*len(files), 'DF': 0, 'IDF': 0, 'TF-IDFs': [0]*len(files), 'TF-IDF-Sum': 0} # making format of the dictionary for every distinct term
                                index[word]['TFs'][count - 1] += 1                 # counting Term Frequency of each term
                                index[word]['DF'] += 1                             # counting Document Frequency of each term
                            else:
                                if(index[word]['TFs'][count - 1] >= 1):            # if term already in index and Term has occured previously in this document
                                    index[word]['TFs'][count - 1] += 1             # increase TF only
                                else:
                                    index[word]['TFs'][count - 1] += 1             # else if Term is occuring first time in this document
                                    index[word]['DF'] += 1                         # increase TF and DF both
                    count += 1
                for i in index:
                    index[i]['IDF'] = math.log(index[i]['DF'], 10)/len(files)   # calculating IDF of every term using formula: IDF = log(N/DF)
                for i in index:
                    for r in range(len(files)):
                        index[i]['TF-IDF-Sum'] += index[i]['TF-IDFs'][r]                # calculating sum of TFxIDF each term in dictionary/index
                        index[i]['TF-IDFs'][r] = index[i]['TFs'][r] * index[i]['IDF']   # calculating TFxIDF of each term in dictionary/index
                # for i in index:
                features100 = heapq.nlargest(100, index[i]['TF-IDF-Sum'].items(), key=lambda i: i[1])
                print(features100)
                if (content == "course"):
                    self.index_Course = index
                    self.features100C = features100
                elif (content == "non-course"):
                    self.index_NonCourse = index
                    self.features100NC = features100
        
        os.chdir(os.path.join(os.getcwd(), ".."))                          # return to the "Assignment 3" directory 
        self.saveIndex(self.index_Course, "index_Course")                  # saving to file after creating index 
        self.saveIndex(self.features100C, "feature100C")
        self.saveIndex(self.index_NonCourse, "index_NonCourse")
        self.saveIndex(self.features100NC, "feature100NC")

    def saveIndex(self, index, name):                                      # saving index to json
        print("\nSaving Index to json file...\n")
        with open('files/' + name + '.json', 'w') as file:                 # open json file as write-mood
            json.dump(index, file, indent=4)                               # dump/store index to json file
        print("\nIndex saved successfully!\n")
        file.close()                                                       # close file

    def loadIndex(self, index, name):                                      # loading index from file to dictionary
        print("\nLoading Index from json file...\n")    
        with open('files/' + name + '.json', 'r') as file:                 # open json file as read-only
            index = json.load(file)                                        # load index to dictonary in python
        print("Index loading successful!\n")
        file.close()                                                       # close file

    def loadORcreateINDEXES(self):
        if(len(self.index_Course) or len(self.index_NonCourse) or len(self.features100C) or len(self.features100NC) == 0):
            if (exists('files/index_Course.json') and (os.path.getsize('files/index_Course.json') == 0)) or \
            (exists('files/index_NonCourse.json') and (os.path.getsize('files/index_NonCourse.json') == 0)) or \
            (exists('files/features100C.json') and (os.path.getsize('files/features100C.json') == 0)) or \
            (exists('files/features100NC.json') and (os.path.getsize('files/features100NC.json') == 0)):
                self.createIndex()
            else:
                if exists('files/index_Course.json') and (os.path.getsize('files/index_Course.json') != 0):
                    self.loadIndex(self.index_Course, "index_Course")                                                 # if index alreading exists in file, load from it
                if exists('files/index_NonCourse.json') and (os.path.getsize('files/index_NonCourse.json') != 0):
                    self.loadIndex(self.index_NonCourse, "index_NonCourse")                                           # if index alreading exists in file, load from it
                if exists('files/features100C.json') and (os.path.getsize('files/features100C.json') != 0):
                    self.loadIndex(self.features100NC, "features100C")
                if exists('files/features100NC.json') and (os.path.getsize('files/features100NC.json') != 0):
                    self.loadIndex(self.features100NC, "features100NC")

model = FeatureSelection()
model.loadORcreateINDEXES()