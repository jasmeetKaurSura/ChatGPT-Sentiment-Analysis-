# ChatGPT-Sentiment-Analysis-
Sentiment analysis of ChatGPT Tweets using VADER and RoBERTa

ChatGPT is an artificial intelligence chatbot developed by OpenAI. Soon after its release in November 2022, ChatGPT has become the talk of the town for its detailed responses and articulated answers across various domains of knowledge. Many people and organizations are speaking in favour of this artificial intelligence. However, not everyone is a fan. People are sharing their thoughts about this AI tool on different social media platforms like Twitter and more. Twitter is well known for being a platform where users can tweet about their emotions. In this regard, sentiment analysis can be used to determine whether a tweet is expressing positive, neutral, or negative emotions. To compare the results of two different models, in this project, sentiment analyses are performed using the VADER and ROBERT models on the tweets posted regarding ChatGPT.

Tools and technique used: Python(Pandas, Numpy, NLTK, Transformer), Power Bi (data visualisation tool)

I used the NLTK Python library, since it already has a built-in, pretrained sentiment analyzer called VADER (Valence Aware Dictionary and Sentiment Reasoner).

As VADER is pretrained, you can get results more quickly than with many other analyzers. VADER is best suited for the short sentences, slang, and abbreviations seen in social media. Although it is less reliable when grading larger, more organized sentences, it is frequently a useful starting point.

For comparison, I used RoBERTa (Robustly Optimized BERT-Pretraining Approach), which pre-trained moddel by hugging faces. RoBERTa is trained on a massive dataset that spans over 160GB of uncompressed text. RoBERTa is trained for longer sequences, but it is really time-consuming.
