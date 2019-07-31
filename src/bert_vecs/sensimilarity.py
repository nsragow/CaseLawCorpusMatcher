from io import BytesIO
import numpy as np
from SQLPanda import lite_load
from sklearn.neighbors import NearestNeighbors
from bert_serving.client import BertClient
bc = BertClient()
#rets = bc.encode(sen_list[ind:ind+batch_size])


def convert_to_array(b_string):
    b = BytesIO()
    b.write(b_string)

    b.seek(0)

    return np.load(b)

class PrecompiledModel():
    def __init__(self, path_to_sqlite , embedding_function = None, offset = 4500):
        self.offset = offset
        print("Running SQL Query...               ",end="\r")
        sdf = lite_load(path_to_sqlite)
        self.df = sdf.q(f"select ndarray, sentence from senvecs")
        print("Converting to ndarrays...               ",end="\r")
        self.converted = self.df.ndarray.apply(convert_to_array)
        X = list(self.converted.iloc[offset:].values)
        print("Building Nearest Neighbor model...               ",end="\r")
        self.nn = NearestNeighbors(n_neighbors=3, algorithm='auto',metric='cosine', n_jobs=-1)
        self.nn.fit(X)
        print("Done!                                     ")
    def predict(self, sentence):
        rets = bc.encode([sentence])
        top_res = self.nn.kneighbors([rets[0]])[1][0]
        top_res_sentences = []
        for i in top_res:
            top_res_sentences.append(self.df.sentence.iloc[self.offset+i])
        return top_res_sentences
