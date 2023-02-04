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


dataset_file='/tmp/pycharm_project_631/docs/gen_docs_func_all_top_sen_ner_manual_covid_bert.csv'
def get_embeddings(dataset_file):
    '''

    :param dataset_file: it should have h_sentence column
    :return:
    '''

    data_df=pd.read_csv(dataset_file,sep=';')

    data_df.dropna(inplace=True)
    h_embeddings=[]
    final_sens=[]
    for ii,data_row in tqdm(data_df.iterrows(),total=data_df.shape[0]):

        doc_sens = [(i.split('\t')[0],float(i.split('\t')[1])) for i in data_row['top_sens'].split(",") if len(i)>0]
        doc_sens_sorted = sorted(doc_sens, key=lambda t: t[1], reverse=True)
        doc_sens=[i[0] for i in doc_sens_sorted if i[1]>0.07]
        final_sens.append(doc_sens)
        if len(doc_sens)>0:
            sen_embeddings=model.encode(doc_sens)
            sen_embeddings=[i.tolist() for i in sen_embeddings]
            h_embeddings.append(sen_embeddings)

        else:
            h_embeddings.append([])

    data_df['h_embeddings']=h_embeddings
    data_df['sentences']=final_sens

    target_file=f'''{".".join(dataset_file.split(".")[:-1])}-embed.csv'''

    data_df.to_csv(target_file,sep=';',index=None) #h_embeddings

get_embeddings(dataset_file)