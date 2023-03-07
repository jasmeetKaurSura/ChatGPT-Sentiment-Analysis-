
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import sys 


# Sentiment Analysis using vader analysis
def Vader(df):
    sia = SentimentIntensityAnalyzer()
    Polarity_Score=[]
    for row in tqdm(df['Clean_tweet'], desc= "vader sentiment analysis", total= len(df)):
        x = sia.polarity_scores(row)
        Polarity_Score.append(x)
    sentiment_df= pd.DataFrame(Polarity_Score)
   
    df=pd.concat([df.reset_index(drop=True),sentiment_df],axis=1)
    
    df['sentiment']= df['compound'].apply(lambda score: 'positive' if score>0 else 'negative' if score<0 else 'neutral')
    df['keywords']= df['tweets'].apply(lambda words: words.split(" "))
    return(df)


