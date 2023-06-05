import json
import pandas as pd
import numpy as np
import re
import sys
import nltk
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import collections
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('MAIN.csv')

df['remove_lower_punct'] = df['abstract'].str.lower().str.replace("'", '').str.replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()

print(df.head(10))

# dont use VADER use BERT for sentiment anlasyis

analyser = SentimentIntensityAnalyzer()

sentiment_score_list = []
sentiment_label_list = []

for i in df['remove_lower_punct'].values.tolist():
    sentiment_score = analyser.polarity_scores(i)

    if sentiment_score['compound'] >= 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Positive')
    elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Neutral')
    elif sentiment_score['compound'] <= -0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Negative')

df['sentiment'] = sentiment_label_list
df['sentiment score'] = sentiment_score_list

print(df.head(10))

df['tokenise'] = df.apply(lambda row: nltk.word_tokenize(row['remove_lower_punct']), axis=1)
stop_words = set(stopwords.words("english"))
df['remove_stopwords'] = df['tokenise'].apply(lambda x: [item for item in x if item not in stop_words])
wordnet_lemmatizer = WordNetLemmatizer()
df['lemmatise'] = df['remove_stopwords'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])


vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))

vectors = []

for index, row in df.iterrows():
    vectors.append(", ".join(row[6]))

vectorised = vectorizer.fit_transform(vectors)

print(vectorised)


# used LDA Model here use the previous given code instead

lda_model = LatentDirichletAllocation(n_components = 5, # number of topics
                                  random_state = 10,          # random state
                                  evaluate_every = -1,      # compute perplexity every n iters, default: Don't
                                  n_jobs = -1,              # Use all available CPUs
                                 )
lda_output = lda_model.fit_transform(vectorised)
topic_names = ["Topic" + str(i) for i in range(1, lda_model.n_components + 1)]
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns = topic_names)
dominant_topic = (np.argmax(df_document_topic.values, axis=1)+1)
df_document_topic['Dominant_topic'] = dominant_topic
df = pd.merge(df, df_document_topic, left_index = True, right_index = True, how = 'outer')


docnames = ['Doc' + str(i) for i in range(len(df_document_topic))]
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topic_names, index=docnames)
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic
df_topic_keywords = pd.DataFrame(lda_model.components_)
df_topic_keywords.columns = vectorizer.get_feature_names_out()
df_topic_keywords.index = topic_names
df_topic_no = pd.DataFrame(df_topic_keywords.idxmax())
df_scores = pd.DataFrame(df_topic_keywords.max())
tmp = pd.merge(df_topic_no, df_scores, left_index=True, right_index=True)
tmp.columns = ['topic', 'relevance_score']
print(tmp)

all_topics = []

for i in tmp['topic'].unique():
    tmp_1 = tmp.loc[tmp['topic'] == i].reset_index()
    tmp_1 = tmp_1.sort_values('relevance_score', ascending=False).head(1)

    tmp_1['topic'] = tmp_1['topic']

    tmp_2 = []
    tmp_2.append(tmp_1['topic'].unique()[0])
    tmp_2.append(list(tmp_1['index'].unique()))
    all_topics.append(tmp_2)

all_topics = pd.DataFrame(all_topics, columns=['Dominant_topic', 'topic_name'])
print(all_topics)


# results = df.groupby(['Dominant_topic', 'sentiment']).count().reset_index()
# results = results.merge(all_topics, on='Dominant_topic')
# results['topic_name'] = results['topic_name'].apply(', '.join)
# graph_results = results[['topic_name', 'sentiment', 'sentiment score']]
# graph_results = graph_results.pivot(index='topic_name', columns='sentiment', values='sentiment score').reset_index()
# graph_results.set_index('topic_name', inplace=True)
# print(graph_results)
# fig = graph_results.plot.bar(rot=90, figsize=(10,10))
# fig.figure.savefig('sentiment_analysis.png', bbox_inches='tight')

# pip install transformers[sentencepiece]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline

absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification \
  .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
                          tokenizer=sentiment_model_path)
sentence = "The AAA is doing a good job of putting people into boxes based on race and other characteristics. I do not consider this to be a good thing. I believe diversity of thought and opinion is a good thing, but measuring diversity based on how people look is ridiculous."
print(f"Sentence: {sentence}")
print()

aspect = "diversity of thought and opinion"
inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
outputs = absa_model(**inputs)
probs = F.softmax(outputs.logits, dim=1)
probs = probs.detach().numpy()[0]
print(f"Sentiment of aspect '{aspect}' is:")
for prob, label in zip(probs, ["negative", "neutral", "positive"]):
  print(f"Label {label}: {prob}")
print()


aspect = "diversity based on how people look"
inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
outputs = absa_model(**inputs)
probs = F.softmax(outputs.logits, dim=1)
probs = probs.detach().numpy()[0]
print(f"Sentiment of aspect '{aspect}' is:")
for prob, label in zip(probs, ["negative", "neutral", "positive"]):
  print(f"Label {label}: {prob}")
print()

sentiment = sentiment_model([sentence])[0]
print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")
