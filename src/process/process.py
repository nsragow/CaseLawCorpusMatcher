import json
from io import StringIO
import xml.etree.ElementTree as ET
import numpy as np
from spacy.lang.en import English
import spacy

nlp = English()

def get_by_index(filename,index):
    with open(filename,"r") as fp:
        for i, line in enumerate(fp):
            if i == index:
                return json.loads(line)
                break



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
    def __init__(self,filename):
        self.filename = filename
        self._init_lines_()
    def _init_lines_(self):
        self.line_count = 0
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                self.line_count += 1
    def _map_line_(self,function):
        returns = np.ndarray(self.line_count)
        with open(self.filename,"r") as fp:
            for i, line in enumerate(fp):
                if (i % 100) == 0:
                    print(f"finished {i}/{self.line_count}            ",end="\r")
                returns[i] = function(line)
        return returns
    def map(self,function):

        new_func = lambda raw : function(json.loads(raw))
        return self._map_line_(new_func)
    def text_map(self,function):
        new_func = lambda raw : function(get_decision(raw))
        return self.map(new_func)
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
