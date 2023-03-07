import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
import sys

# download these packages in nltk if you dont have it already 
# nltk.download(["names","stopwords","twitter_samples", "averaged_perceptron_tagger","vader_lexicon","punkt"])


#create data for stopwords like a, the and I etc 
stopwords = nltk.corpus.stopwords.words("english")
Clean_tweet=[]
def clean(df):
    punc = '''!(-[{;:'"\/|<>,.?@#$%^&*_~`}])'''
    i =0
    for row in tqdm(df['tweets'], desc="cleaning", total=len(df)):
        row = re.sub(r'@[A-Za-z0-9]+', '', row) # remove @menstions
        row = re.sub(r'#', '', row) # remove #
        row = re.sub(r'\n', '',row) 
        row = re.sub(r'RT[\s]+', '', row) # remove RT
        row = re.sub(r'https?:\/\/\S+', '', row)
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
        row= emoji_pattern.sub(r'', row) # no emoji

        for word in row:
                for char in word:
                    if char in punc:
                        row = row.replace(char, " ")
            
        text_tokens = word_tokenize(row.lower())
        tokens_without_sw = [word for word in text_tokens if word not in stopwords]
        row= (" ").join(tokens_without_sw)
        

          
        Clean_tweet.append(row)
        i+= 1
    Clean_tweet_df =pd.DataFrame(Clean_tweet, columns= ["Clean_tweet"])
    df=pd.concat([df.reset_index(drop=True),Clean_tweet_df],axis=1)
    return df
