from transformers import pipeline
pipe = pipeline("text-classification", model="publichealthsurveillance/PHS-BERT", device=0)

def check_sentence(sens):
    final_sen=[]
    for sen in sens:
        if len(sen.split())>4:
            label_list=pipe(sen)[0]
            label_text=label_list['label']
            label_score=label_list['score']
            if label_text=='LABEL_1' and label_score>0.40:
                final_sen.append(sen)
    return final_sen
