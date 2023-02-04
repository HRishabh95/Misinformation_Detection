import pandas as pd
import ast
target_file='/tmp/pycharm_project_631/docs/gen_docs_func_all_top_sen_ner_manual_covid_bert-embed_5-facts.csv'
# target_file='./recovery-news-data-cleaned-facts.csv'
scores=pd.read_csv(target_file,sep=';')

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

# for ii, i in enumerate(final_score):
#     cred = 0
#     for jj,j in enumerate(i):
#         if jj < int(len(i) * 0.3):
#             if j == 5:
#                 cred += 1
#     percent.append([cred / int(len(final_score[ii]) * 0.3), int(len(final_score[ii]))])
