import pandas as pd
from pymed import PubMed
import json
wic_data = '/tmp/pycharm_project_631/all_journal_content_cite.csv'
df_docs = pd.read_csv(wic_data, sep='\t', index_col=0, lineterminator='\n').dropna(subset=['contents'])

pubmed = PubMed(tool="MyTool", email="uhrishabh@gmail.com")

citations=[]
for ii,df_rows in df_docs.iterrows():
    if len(df_rows['citation'])==2:
        print(ii)
        results = pubmed.query(df_rows['title'], max_results=1)
        for result in results:
            dd=json.loads(result.toJSON())
        authors=dd['authors']
        journal_authors=''
        for author in authors:
            journal_authors+=', '+author['lastname']+author['firstname']
        year=dd['publication_date'].split("-")[0]
        journal=dd['journal']
        title=df_rows['title']
        pubmed_id=dd['pubmed_id'].split("\n")[0]
        url = f'''https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}'''
        apa_citation = f'''{journal_authors} ({year}).{title}.{journal},{0},{0}, {url}'''
        citations.append([df_rows['id'],apa_citation])

