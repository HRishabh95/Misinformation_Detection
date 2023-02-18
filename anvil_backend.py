import modin.pandas as pd
import numpy as np
import faiss
import torch
import anvil.server
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('GPL/trec-covid-v2-msmarco-distilbert-gpl')
model = model.to(torch.device("cuda"))

#if torch.cuda.is_available():
anvil.server.connect("server_B4B5VFLKQXFTJNUYILIXP4FX-3KQLWOLABXKEOIWU")

sen_index_df = pd.read_csv("./all_journal_sentences.v2.6.csv", sep=';', engine='python')

index_filename = './journal_index.v2.6.faiss'
index=faiss.read_index(index_filename)
# res = faiss.StandardGpuResources()
# gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
# index = index


@anvil.server.callable
def get_facts(claim):
    sen_embeddings = model.encode(claim)
    score,ids = index.search(sen_embeddings.reshape(1,-1), k=20)
    ids=ids[0]
    score=score[0]
    score_text = []
    d_sen=[]
    for j in zip(ids, score):
        if j[1]<40:
            sen = sen_index_df.iloc[[j[0]-2,j[0]-1,j[0]]]
            sen_text=" ".join([i['text'] for ii,i in sen.iterrows() if i['text']])
            #citation=sen['citation']
            if sen_text not in d_sen:
                evidence_emb=model.encode(sen_text)
                cosine = np.dot(sen_embeddings, evidence_emb) / (norm(sen_embeddings) * norm(evidence_emb))
                if cosine>=0.8:
                    d_sen.append(sen_text)
                    score_text.append([sen_text,j[1]])
    return np.asarray(score_text)

anvil.server.wait_forever()
