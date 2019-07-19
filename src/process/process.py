import json
from io import StringIO
import xml.etree.ElementTree as ET
import numpy as np
from spacy.lang.en import English
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack
from sklearn.neighbors import NearestNeighbors

nlp = English()

def get_by_index(filename,index):
    with open(filename,"r") as fp:
        for i, line in enumerate(fp):
            if i == index:
                return json.loads(line)
                break

def add_to_set(word_list,word_set):
    for word in word_list:
        if type(word) == str:
            word_set.add(word.lower())
        else:
            raise Exception("Expected String got "+word+ " of type " + str(type(word)))


def get_decision(json):
    body = json["casebody"]["data"]

    tree = ET.parse(StringIO(body))
    decision = tree.findall("{http://nrs.harvard.edu/urn-3:HLS.Libr.US_Case_Law.Schema.Case_Body:v1}opinion[@type='majority']/{http://nrs.harvard.edu/urn-3:HLS.Libr.US_Case_Law.Schema.Case_Body:v1}p")
    return "\n".join([line.text for line in decision])

def spacify(text):
    filtered_sent=[]

    #  "nlp" Object is used to create documents with linguistic annotations.
    doc = nlp(text)

    # filtering stop words
    for word in doc:

        if (word.is_stop==False) and (word.is_punct == False) and (word.is_digit == False):
            filtered_sent.append(word.lemma_)

    return filtered_sent

class StateCorpus:
    def build_dictionary(self):
        if self.dictionary is None:
            self.dictionary = set()
            self.spacied_map(lambda spac : add_to_set(spac,self.dictionary))
            self.vectorizer = CountVectorizer(vocabulary=list(self.dictionary),input="content")
    def bag_of_words(self):

        self.build_dictionary()

        return vstack(self.spacied_map(lambda words : self.vectorizer.transform([" ".join(words)])))
    def build_nn(self):
        if self.nn is None:
            self.build_dictionary()
            self.nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            self.nn.fit(self.bag_of_words())
    def predict(self,unclean_text, count=3):
        clean_text = get_decision(json.loads(unclean_text))

        vectorized = self.vectorizer.transform([" ".join(spacify(clean_text))])

        nearest_neighbors_index = self.nn.kneighbors(vectorized,n_neighbors=count,return_distance=True)[1][0]
        
        nearest_neighbors = [self.text(index) for index in nearest_neighbors_index]

        return (clean_text,nearest_neighbors)



    def __init__(self,filename,filelimit = -1):
        self.nn = None
        self.dictionary = None
        self.filename = filename
        self._init_lines_(filelimit)
    def _init_lines_(self,filelimit):
        self.line_count = 0
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                self.line_count += 1
        if filelimit > 0:
            self.line_count = min(self.line_count,filelimit)
    def _map_line_(self,function):
        returns = list()
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                if i > self.line_count:
                    break
                if (i % 100) == 0:
                    print(f"finished {i}/{self.line_count}            ",end="\r")
                returns.append(function(line))
        return returns
    def map(self,function):

        new_func = lambda raw : function(json.loads(raw))
        return self._map_line_(new_func)
    def text_map(self,function):
        new_func = lambda raw : function(get_decision(raw))
        return self.map(new_func)
    def _raw_get_(self,index):

        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                if i == index:
                    return line
                    break
    def get(self,index):
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                if i == index:
                    return json.loads(line)
                    break
    def text(self,index):
        return get_decision(self.get(index))
    def spacied(self,index):
        return spacify(self.text(index))
    def spacied_map(self,function):
        new_func = lambda text : function(spacify(text))
        return self.text_map(new_func)
    def __iter__(self):

        return CorpusIterator(self.filename)
    def __next__(self):
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                if (i % 100) == 0:
                    print(f"finished {i}/{self.line_count}            ",end="\r")
                returns[i] = function(line)
        return returns
class CorpusIterator:
    def __init__(self,filepath):
        self.file = open(filepath,"r")
    def __next__(self):

        line = self.file.readline()
        if line:
            return [spacify(get_decision(json.loads(line)))]
        else:
            self.file.close()
            raise StopIteration
