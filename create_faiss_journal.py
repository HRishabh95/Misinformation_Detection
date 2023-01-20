import os.path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
model = SentenceTransformer('GPL/trec-covid-v2-msmarco-distilbert-gpl')
# Encoding: bert-base-nli-mean-tokens
# pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
print(model.device)
import re

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "..." in text: text = text.replace("...", "<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


import string

PUNCTUATIONS = string.punctuation.replace(".", "")


def remove_punctuation(text):
    trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
    return text.translate(trans)


def remove_whitespaces(text):
    return " ".join(text.split())


def clean_en_text(text):
    """
    text
    """
    text = re.sub(r"[^0-9a-zA-Z(),!?\'`]", " ", text)
    # text= re.sub(r'\d+', '',text)
    text = remove_punctuation(text)
    text = remove_whitespaces(text)
    return text.strip().lower()


if not os.path.isfile('./all_journal_sentences.csv'):
    wic_data = '/tmp/pycharm_project_631/all_journal_content.csv'
    df_docs = pd.read_csv(wic_data, sep='\t', index_col=0, lineterminator='\n').dropna()
    # df_docs['contents'] = df_docs.apply(lambda x: clean_en_text(x['contents']), axis=1)
    df_docs.columns = ['title', 'text', 'citation', 'views', 'j_type', 'docno']

    df_docs['sentences'] = df_docs.apply(lambda x: split_into_sentences(x['text']), axis=1)

    sen_index = []
    for ii, df_rows in df_docs.iterrows():
        doc_ids = df_rows['docno']
        for sid, sen in enumerate(df_rows['sentences']):
            if len(sen.split()) > 4:
                sen_ids = f'''{doc_ids}_s{sid}'''
                sen_index.append([doc_ids, sen_ids, sen])

    sen_index_df = pd.DataFrame(sen_index, columns=['docno', 'sid', 'text'])
    sen_index_df.to_csv('./all_journal_sentences.csv', index=None, sep=';')
else:
    sen_index_df = pd.read_csv("./all_journal_sentences.csv", sep=';', engine='python')

sen_index_df.dropna(inplace=True)
embeddings = model.encode(sen_index_df['text'].to_list(), show_progress_bar=True)

import numpy as np
import faiss

embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index = faiss.IndexIDMap(index)
index.add_with_ids(embeddings, sen_index_df.index.values)
index_filename = './journal_index.faiss'
faiss.write_index(index, index_filename)
