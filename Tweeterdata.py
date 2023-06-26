import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

data = pd.read_excel("/Users/umayyentur/Project/Tweet/LabeledText.xlsx")
data1 = data.copy()

tweet = pd.DataFrame(data1["Caption"])

tweet_words = []

for caption in tweet["Caption"]:
    for word in caption.split(" "):
        if word.startswith("@") and len(word) > 1:
            word = "@user"
        elif word.startswith("http"):
            word = "http"
        tweet_words.append(word)

tweet_proc = " ".join(tweet_words)
print(tweet_proc)

# Model and tokenizer download
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ["Negative", "Neutral", "Positive"]

# Sentiment analysis
encoded_tweet = tokenizer.encode_plus(tweet_proc, return_tensors="pt", padding=True, truncation=True, max_length=512)
output = model(**encoded_tweet)

scores = output.logits.detach().numpy()
scores = softmax(scores)

for i in range(len(scores[0])):
    l = labels[i]
    s = scores[0][i]
    print(l, s)
