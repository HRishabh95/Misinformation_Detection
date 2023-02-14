import pandas as pd
import time
from semanticscholar import SemanticScholar
sch = SemanticScholar(api_key='chcsWY6XL5a3fp4Vgolsu82vbc629SSP7DIs9MFA')
wic_data = '/tmp/pycharm_project_631/all_journal_content.csv'
df_docs = pd.read_csv(wic_data, sep='\t', index_col=0, lineterminator='\n').dropna()

apa_citations=[]
for ii,df_rows in df_docs.iterrows():
    print(ii)
    try:
        title=df_rows['title']
        results = sch.search_paper(title)
        if results:
            authors=", ".join([i['name'] for i in results[0].authors])
            year=results[0].year
            title=results[0].title
            if results[0].journal:
                journal=results[0].journal.name
                if results[0].journal.pages:
                    page = results[0].journal.pages.strip()
                else:
                    page = '1'
                if results[0].journal.volume:
                    volume = results[0].journal.volume.strip().replace(" ", "(") + ")"
                else:
                    volume = '0'
            else:
                journal='JAMA'
                page = '1'
                volume= '0'
            url=results[0].url
            apa_citation=f'''{authors} ({year}).{title}.{journal},{volume},{page}, {url}'''
            apa_citations.append(apa_citation)
        else:
            apa_citations.append([])
    except:
        print("Error")
        apa_citations.append([])

df_docs['citation']=apa_citations
df_docs.to_csv('/tmp/pycharm_project_631/all_journal_content_cite.csv',sep='\t')