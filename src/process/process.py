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


def write_new_mexico_to_file(json,appendable):

    for entry in json["casebody"]["data"]["opinions"]:
        if entry["type"] == "majority":
            appendable.write(entry["text"].replace("\n"," ").lower()+"\n")
            break

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
    '''
    Case law file handler / model container.
    Use with the data.jsonl (Arkansas) from https://case.law/bulk/download/

    Example
        corp = StateCorpus("/Users/user/path/to/data/Arkansas/data/data.jsonl",100)
        corp.build_nn() # Create model
        text = corp._raw_get_(200) # grabbing a single jsonl line at index 200
        ret = corp.predict(text)
        print(ret[0]) # Cleaned text to predict on. Useful for sanity check
        rank = 0 # most near neighbor
        print(ret[1][rank]) # clean corpus of most near neighbor
    '''
    def build_dictionary(self):
        '''
        Initiates a dictionary and CountVectorizer for all words in the .jsonl file.
        '''
        if self.dictionary is None:
            self.dictionary = set()
            self.spacied_map(lambda spac : add_to_set(spac,self.dictionary))
            self.vectorizer = CountVectorizer(vocabulary=list(self.dictionary),input="content")

    def bag_of_words(self):
        '''
        Returns Bag of Words for each passage
        '''
        self.build_dictionary()

        return vstack(self.spacied_map(lambda words : self.vectorizer.transform([" ".join(words)])))
    def build_nn(self):
        '''
        Initiates NearestNeighbors model
        '''
        if self.nn is None:
            self.build_dictionary()
            self.nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            self.nn.fit(self.bag_of_words())
    def predict(self,unclean_text, count=3):
        '''
        Using NearestNeighbors model, returns the top most similar texts to unclean_text

        Parameters
        ----------
            unclean_text : str
                    Single line in .jsonl file. Predict will clean and extract the appropriate
                    text.
            count : int
                The number of desired nearest neighbors. Should not be larger
                than the number of texts in the model.
        '''
        clean_text = get_decision(json.loads(unclean_text))

        vectorized = self.vectorizer.transform([" ".join(spacify(clean_text))])

        nearest_neighbors_index = self.nn.kneighbors(vectorized,n_neighbors=count,return_distance=True)[1][0]

        nearest_neighbors = [self.text(index) for index in nearest_neighbors_index]

        return (clean_text,nearest_neighbors)



    def __init__(self,filename,filelimit = -1):
        '''
        Constructor

        Parameters
        ----------
            filename : str
                Path to jsonl file downloaded of case.law
            filelimit : int
                Maximum amount of lines to be read from the jsonl. When -1 all
                lines will be read.
        '''
        self.nn = None
        self.dictionary = None
        self.filename = filename
        self._init_lines_(filelimit)
    def _init_lines_(self,filelimit):
        '''
        Counts the amount of lines in the jsonl
        or uses the filelimit.

        Parameters
        ----------
            filelimit : int
                See Constructor
        '''
        self.line_count = 0
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                self.line_count += 1
        if filelimit > 0:
            self.line_count = min(self.line_count,filelimit)
    def _map_line_(self,function):
        '''
        Applies a function to every line (as a str) in the jsonl.
        Similar to pandas.DataFrame.map(axis=1)

        Parameters
        ----------
            function : function
                The function to be applied to each line
        Returns
        -------
            List of the return values of the function in order of lines
        '''
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
        '''
        See _map_line_.

        Instead of mapping to the raw text, maps to the json of the raw text.
        '''

        new_func = lambda raw : function(json.loads(raw))
        return self._map_line_(new_func)
    def text_map(self,function):
        '''
        See _map_line_.

        Instead of mapping to the raw text, maps to the text of the case.
        '''

        new_func = lambda raw : function(get_decision(raw))
        return self.map(new_func)
    def _raw_get_(self,index):
        '''
        Get line as str from jsonl

        Parameters
        ----------
            index : int
                The index of the line starting at 0
        '''
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                if i == index:
                    return line
                    break
    def get(self,index):
        '''
        Get the python dict from json.loads of the line at the index.

        Parameters
        ----------
            index : int
                see _raw_get_
        '''
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                if i == index:
                    return json.loads(line)
                    break
    def text(self,index):
        '''
        Gets the corpus from the line at index.

        see get().
        '''
        return get_decision(self.get(index))
    def spacied(self,index):
        return spacify(self.text(index))
    def spacied_map(self,function):
        '''
        See _map_line_.

        Instead of mapping to the raw text, maps to the corpus after the spacy function is applied.
        '''

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
