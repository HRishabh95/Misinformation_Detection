import pandas as pd
import ast

scores=pd.read_csv('./recovery-news-data-cleaned-facts.csv',sep=';')

final_score=[]
for ii,score in scores.iterrows():
    final_fact_score = []
    facts=ast.literal_eval(score['facts'])
    if facts:
        for fact in facts:
            fact_score=0
            evis=fact.split("\t")
            for evi in evis:
                if evi:
                    try:
                        evi_text,evi_score=";".join(evi.split(";")[:-1]),float(evi.split(";")[-1])
                        if evi_score<35:
                            fact_score+=1
                    except:
                        continue
            final_fact_score.append(fact_score)
        final_score.append(final_fact_score)
    else:
        final_score.append([0])

scores['final_score']=final_score
scores.to_csv('./recovery-news-data-cleaned-facts-passage-score.csv', index=None, sep=';')
