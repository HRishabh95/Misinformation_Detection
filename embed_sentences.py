import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import ast


#model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
model = SentenceTransformer('GPL/trec-covid-v2-msmarco-distilbert-gpl')
#if torch.cuda.is_available():
model = model.to(torch.device("cpu"))
#print(model.device)


dataset_file='./data/train_qid_sen.csv'
def get_embeddings(dataset_file):
    '''

    :param dataset_file: it should have h_sentence column
    :return:
    '''

    data_df=pd.read_csv(dataset_file,sep=';')


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

    target_file=f'''{".".join(dataset_file.split(".")[:-1])}-embed.csv'''

    data_df.to_csv(target_file,sep=';',index=None) #h_embeddings

get_embeddings(dataset_file)