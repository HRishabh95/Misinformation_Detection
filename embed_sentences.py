import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import ast


#model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
model = SentenceTransformer('GPL/trec-covid-v2-msmarco-distilbert-gpl')
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
print(model.device)

data_df=pd.read_csv('./recovery-news-data-cleaned.csv',sep=';')


h_embeddings=[]
for ii,data_row in tqdm(data_df.iterrows(),total=data_df.shape[0]):
    sen_list=ast.literal_eval(data_row['h_sentences'])
    if len(sen_list)>0:
        sen_embeddings=model.encode(sen_list)
        sen_embeddings=[i.tolist() for i in sen_embeddings]
        h_embeddings.append(sen_embeddings)
    else:
        h_embeddings.append([])
data_df['h_embeddings']=h_embeddings
data_df.to_csv('./recovery-news-data-cleaned-embed.csv',sep=';',index=None) #h_embeddings