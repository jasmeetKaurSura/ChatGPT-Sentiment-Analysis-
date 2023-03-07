import pandas as pd

from Analysis import plot
from Clean import clean
from Vader_Sentiment import Vader
from Roberta_Sentiment import Roberta


df =pd.DataFrame
df =pd.read_csv('ChatGPT.csv')
df= df.head(106)
print(df)
df= clean(df)
df= Vader(df)
df =Roberta(df)
print(df)
df.to_csv('test1.csv', index=False)

# plot('test.csv')


