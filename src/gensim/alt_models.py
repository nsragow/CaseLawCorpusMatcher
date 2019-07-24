from gensim.models import Doc2Vec
import multiprocessing

cores = multiprocessing.cpu_count()
min_count = 10
max_vocab_size = None
#DBOW
def dbow():
    return Doc2Vec(max_vocab_size = max_vocab_size,dm=0, size=100, negative=5, min_count=min_count, workers=cores, alpha=0.065, min_alpha=0.065)
#model_ug_dbow = dbow()


#DMC
def dmc():
    return Doc2Vec(max_vocab_size = max_vocab_size,dm=1, dm_concat=1, size=100, window=2, negative=5, min_count=min_count, workers=cores, alpha=0.065, min_alpha=0.065)
#model_ug_dmc = dmc()


#DMM
def dmm():
    return Doc2Vec(max_vocab_size = max_vocab_size,dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=min_count, workers=cores, alpha=0.065, min_alpha=0.065)
#model_ug_dmm = dmm()
'''
#Combined 1 DBOW + DMC
def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

train_vecs_dbow_dmc = get_concat_vectors(model_ug_dbow,model_ug_dmc, x_train, 200)
validation_vecs_dbow_dmc = get_concat_vectors(model_ug_dbow,model_ug_dmc, x_validation, 200)

#Combined 2 DBOW and DMM


train_vecs_dbow_dmm = get_concat_vectors(model_ug_dbow,model_ug_dmm, x_train, 200)
validation_vecs_dbow_dmm = get_concat_vectors(model_ug_dbow,model_ug_dmm, x_validation, 200)
'''
print(Doc2Vec())
