import faiss
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

index_filename = './journal_index.faiss'
index=faiss.read_index(index_filename)

#./recovery-news-data-cleaned-embed.csv './data/train_qid_sen-embed.csv'
data_file='/tmp/pycharm_project_631/docs/gen_docs_func_all_top_sen_ner_manual_covid_bert-embed.csv'
data_df=pd.read_csv(data_file,sep=';')
sen_index_df = pd.read_csv("./all_journal_sentences.csv", sep=';', engine='python')
gpu=True
if gpu:
    res = faiss.StandardGpuResources()
    gpu_index=faiss.index_cpu_to_gpu(res,0,index)
    index=index

facts=[]
for ii,data_rows in tqdm(data_df.iterrows(),total=data_df.shape[0]):
    dds=ast.literal_eval(data_rows['h_embeddings'])
    if dds:
        score_final=[]
        for dd in dds:
            search_result=index.search(np.array([dd]).astype('float32'),k=5)
            ids=search_result[1][0]
            score=search_result[0][0]
            score_text=''
            for j in zip(ids,score):
                score_text+=f'''{sen_index_df.iloc[j[0]]['text']};{j[1]}\t'''
            score_final.append(score_text)
        facts.append(score_final)
    else:
        facts.append([])

data_df['facts']=facts
target_file = f'''{".".join(data_file.split(".")[:-1])}-facts.v1.csv'''
data_df.to_csv(target_file, index=None, sep=';')
