
import pandas as pd
import numpy
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import json
import sys 

# Sentiment Analysis using Roberta Pretrained Model


def Roberta(df):
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
   
    Scores_df= pd.DataFrame(columns= ('R_neg','R_neu', 'R_pos','max'))
    
    i=0
    for row in tqdm(df['tweets'],desc=" RoBERTa Sentiment Analysis", total= len(df),  position=0, leave=True):
        
        encoded_text = tokenizer(row, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        Scores_df.loc[i] = (scores[0], scores[1], scores[2], max(scores[0], scores[1], scores[2]) )
        i +=1
        
    
    df=pd.concat([df.reset_index(drop=True),Scores_df],axis=1)
    
    R_Sentiment =[]
    for i in range(0,len(df), 1):
        print(i)
        if df['max'][i] == df['R_neg'][i]:
            R_Sentiment.append('Negative')
        if df['max'][i]== df['R_pos'][i]:
            R_Sentiment.append('Positive')
        else:
            R_Sentiment.append('Neutral')
    
    sentiment_df= pd.DataFrame(R_Sentiment , columns= ["R_sentiment"])
   
    df=pd.concat([df.reset_index(drop=True),sentiment_df],axis=1)
    

    
    
    return(df)
  