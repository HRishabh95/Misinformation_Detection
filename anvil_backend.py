import modin.pandas as pd
import numpy as np
import faiss
import torch
import anvil.server
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('GPL/trec-covid-v2-msmarco-distilbert-gpl')
model = model.to(torch.device("cuda"))

#if torch.cuda.is_available():
anvil.server.connect("server_B4B5VFLKQXFTJNUYILIXP4FX-3KQLWOLABXKEOIWU")

sen_index_df = pd.read_csv("./all_journal_sentences.v2.csv", sep=';', engine='python')

index_filename = './journal_index.v2.faiss'
index=faiss.read_index(index_filename)
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
index = index


@anvil.server.callable
def get_facts(claim):
    sen_embeddings = model.encode(claim)
    score,ids = index.search(sen_embeddings.reshape(1,-1), k=10)
    ids=ids[0]
    score=score[0]
    score_text = []
    d_sen=[]
    for j in zip(ids, score):
        if j[1]<25:
            sen = sen_index_df.iloc[j[0]]['text']
            if sen not in d_sen:
                d_sen.append(sen)
                score_text.append([sen,j[1]])

    return score_text