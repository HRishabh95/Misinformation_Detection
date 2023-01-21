import faiss
import pandas as pd
import ast
index_filename = './journal_index.faiss'

index=faiss.read_index(index_filename)

data_df=pd.read_csv('./recovery-news-data-cleaned-embed.csv',sep=';')

#index.search(np.array([dd[0]]).astype('float32'),k=3)
sen_index_df = pd.read_csv("./all_journal_sentences.csv", sep=';', engine='python')
