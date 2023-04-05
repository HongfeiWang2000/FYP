
import numpy as np 
import pandas as pd
import os
import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, auc
import nltk
# Uncomment to download "stopwords"
nltk.download("stopwords")
from nltk.corpus import stopwords


import datetime
today=datetime.date.today()
formatted_today=today.strftime('%y%m%d')



files = os.listdir('./Datacollect/train')
df = pd.DataFrame()
for file in files:
    df1 = pd.read_csv('Datacollect/train/'+file)
    df = pd.concat([df, df1], ignore_index=True)




df=df.rename(columns={"cat_pro": "target"})


df.target.value_counts()

# Preprocessing
import string
def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)
def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

stop = set(stopwords.words("english"))
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

pattern = re.compile(r"https?://(\S+|www)\.\S+")
for t in df.text:
    matches = pattern.findall(t)
    for match in matches:
        print(t)
        print(match)
        print(pattern.sub(r"", t))
    if len(matches) > 0:
        break

df["text"] = df.text.map(remove_URL) 
df["text"] = df.text.map(remove_punct)
df["text"] = df.text.map(remove_stopwords)



from collections import Counter

# Count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(df.text)
len(counter)
num_unique_words = len(counter)




# Split dataset into training and validation set
train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

# split text and labels
train_sentences = train_df.text.to_numpy()
val_sentences = val_df.text.to_numpy()





from sklearn import preprocessing
from keras.utils import to_categorical
label = preprocessing.LabelEncoder()
y = label.fit_transform(df['target'])
y = to_categorical(y)
print(y)





train_labels = y[:train_size]
train_labels = train_labels.astype(np.uint8)
val_labels = y[train_size:]
val_labels = val_labels.astype(np.uint8)



print(len(train_labels[0]))
print(len(val_labels[0]))



labels = label.classes_
print(labels)



train_sentences.shape, val_sentences.shape




# Tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentences) # fit only to training
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
len(train_sequences)



countmax=0
for i in train_sequences:
    if len(i)> countmax:
        countmax=len(i)
print(countmax)
countmin=100000
for i in train_sequences:
    if len(i)< countmin:
        countmin=len(i)
print(countmin)




bar=1000
kounter=0
for i in train_sequences:
    if len(i)< bar:
        kounter=kounter+1
print(kounter)




# Max number of words in a sequence
max_length = bar

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
train_padded.shape, val_padded.shape




reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])




def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])





from tensorflow.keras import layers
from tensorflow import keras
# Create LSTM model
model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 100, input_length=max_length))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, dropout=0.4, recurrent_dropout=0.4)))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(6,activation='softmax'))
model.summary()





loss = 'categorical_crossentropy'
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy',keras.metrics.Precision(), keras.metrics.Recall()]
model.compile(loss=loss, optimizer=optim, metrics=metrics)





history=model.fit(train_padded, train_labels, epochs=5,validation_data=(val_padded, val_labels))

model.save('saved_model/my_model_1')

import pickle
# saving
with open('./saved_model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



