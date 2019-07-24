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
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if (i % 100) == 0:
                print(f"proc {i}                           ",end="\r")
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(train_path))
test_corpus = list(read_corpus(test_path, tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=10)


model.build_vocab(train_corpus)

#model.train(train_corpus, total_examples=model.corpus_count, epochs=1)

model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])


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




collections.Counter(ranks)  # Results vary between runs due to random seeding and very small corpus



print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


fname = None
model.save(fname)


iters = 10
last_sec = time.time()
times = []
print(f"fin 0/{iters} || 0 elapsed last iteration                ",end="\r")
for i in range(iters):

    #model.train(train_corpus, total_examples=model.corpus_count, epochs=1)
    elapsed = time.time() - last_sec
    times.append(elapsed)
    last_sec = time.time()
    print(f"fin {i+1}/{iters} || {elapsed} elapsed last iteration                ",end="\r")
print("\n")
print("done")
