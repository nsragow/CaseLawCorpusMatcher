import gensim
import os
import collections
import smart_open
import random
import time

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if (i % 100) == 0:
                print(f"proc {i}                           ",end="\r")
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def prep_model(model, path_to_data):
    print("Reading data...                      ", end="\r")
    train_corpus = list(read_corpus(path_to_data))
    print("Building Vocab...                      ", end="\r")
    model.build_vocab(train_corpus)
    print("                                       ", end="\r")
    return train_corpus

def train(model,data,epochs = 1):
    model.train(data, total_examples=model.corpus_count, epochs=epochs)

def evaluate(model,train_corpus):
    ranks = []
    second_ranks = []
    total_docs = len(train_corpus)
    for doc_id in range(200):

        print(f"finished {doc_id}/{total_docs}          ",end="\r")
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])


        print("                                       ", end="\r")

    print(collections.Counter(ranks))
