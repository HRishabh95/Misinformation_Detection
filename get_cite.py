import pandas as pd
import time
from semanticscholar import SemanticScholar
sch = SemanticScholar()
wic_data = '/tmp/pycharm_project_631/all_journal_content.csv'
df_docs = pd.read_csv(wic_data, sep='\t', index_col=0, lineterminator='\n').dropna()

apa_citations=[]
for ii,df_rows in df_docs.iterrows():
    title=df_rows['title']
    results = sch.search_paper(title)
    if results:
        authors=", ".join([i['name'] for i in results[0].authors])
        year=results[0].year
        title=results[0].title
        journal=results[0].journal.name
        page=results[0].journal.pages.strip()
        volume=results[0].journal.volume.strip().replace(" ","(")+")"
        apa_citation=f'''{authors} ({year}).{title}.{journal},{volume},{page}'''
        apa_citations.append(apa_citation)
    else:
        apa_citations.append([])
    time.sleep(5)