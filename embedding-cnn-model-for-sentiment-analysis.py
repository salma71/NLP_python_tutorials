#!/usr/bin/env python
# coding: utf-8

# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Tutorial Overview
# 
# This tutorial is divided into the following parts:
# 1. Movie Review Dataset
# 2. Data Preparation
# 3. Train CNN With Embedding Layer 
# 4. Evaluate Model

# ## 1. Movie Review Dataset
# 
# In this tutorial, I will use the Movie Review Dataset. The Movie Review Data is a collection of movie reviews retrieved from the imdb.com website in the early 2000s by Bo Pang and Lillian Lee. The reviews were collected and made available as part of their research on natural language processing. The reviews were originally released in 2002, but an updated and cleaned up version was released in 2004, referred to as v2.0. The dataset is comprised of 1,000 positive and 1,000 negative movie reviews drawn from an archive of the rec.arts.movies.reviews newsgroup hosted at IMDB. The authors refer to this dataset as the polarity dataset.

# ## Load dataset

# In[13]:


# open the file as read only file
filename = '../input/movie-review/movie_reviews/movie_reviews/neg/cv000_29416.txt'
file = open(filename, 'r')
text = file.read()
# file.close()


# The above snippet loads the texts as an ASCII document where any white spaces are still there. So, we can refactor that into a more generic function to load any document that would help in future imports. 

# In[14]:


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
#     file.close()
    return text


# We have two directories, neg and pos each of which has a text files that need to be read. So, the function `load_doc()` will be responsible for doing that for us. We will getting each file from each directory through the `listdir()`. I will do it in another function to make it more generic.

# In[15]:


from os import listdir

def load_dir(directory):
    for filename in listdir(directory):
        #skip files that doesn't have the right extension
        if not filename.endswith('.txt'):
            next
        # create the full path of the file to open
        path = directory + '/' + filename
        # load document
        doc = load_doc(path)
        print('loading %s' % filename)
        
directory = '../input/movie-review/movie_reviews/movie_reviews/neg'
load_dir(directory)


# ## Clean Dataset
# 
# In this section, we will look at what data cleaning we might want to do to the movie review data. We will assume that we will be using a bag-of-words model or perhaps a word embedding that does not require too much preparation.

# The first step I would like to take is to look at a sample of files to see what sort of words included in the text file. Doing so, would make it more easier to plan ahead which cleaning approach should be taken. I will use the function I did before `load_doc()` then apply the `split()` function to it. 

# In[16]:


filename = '../input/movie-review/movie_reviews/movie_reviews/neg/cv000_29416.txt'
tokens = load_doc(filename).split()
print(tokens)


# From the results, I can see that I need to remove 
# 1. Punctuations, 
# 2. numbers, 
# 3. One word character
# 4. English stopping words
# 5. Punctuation words 
# 
# I will put those all together in a function called `clean_doc()`

# In[17]:


from nltk.corpus import stopwords
import string
import re

def clean_doc(doc):
    """
    input: file name in a text format
    what it does: clean up the passing document from punctuation, stop words,...etc
    output: cleaned document saved in a text file
    """
    #split the given document into tokens 
    tokens = doc.split()
    # prepare the the regex to filter out the punctuation from tokens
    punc_filter = re.compile('%s' %re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [punc_filter.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# In[18]:


filename = '../input/movie-review/movie_reviews/movie_reviews/neg/cv000_29416.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens[:20])


# ## Define a vocabulary
# 
# It is important to define a vocabulary of known words when using a text model. The more words, the larger the representation of documents, therefore it is important to constrain the words to only those believed to be predictive. 
# 
# We can build a vocabulary as a `Counter` from the `collections` library, which is a dictionary mapping of words and their count that allows us to easily update and query. Each document can be added to the counter (a new function called `add_doc_to_vocab()`) and we can step over all of the reviews in the negative directory and then the positive directory (a new function called `load_dir()`). 

# In[19]:



import string
import re
from os import listdir
from collections import Counter 
from nltk.corpus import stopwords

# load doc into memory
def load_doc(filename):
# open the file as read only 
    file = open(filename, 'r')
# read all text
    text = file.read()
    # close the file 
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
# split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab): 
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
      # update counts
    vocab.update(tokens)
    
    
# load all docs in a directory
def process_docs(directory, vocab):
# walk through all files in the folder 
    for filename in listdir(directory):
    # skip any reviews in the test set
        if filename.startswith('cv9'): 
            continue
        # create the full path of the file to open
        path = directory + '/' + filename 
        # add doc to vocab 
        add_doc_to_vocab(path, vocab)
        
        
# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('../input/movie-review/movie_reviews/movie_reviews/pos', vocab) 
process_docs('../input/movie-review/movie_reviews/movie_reviews/neg', vocab) 
# print the size of the vocab 
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(100))


# Running the snippet above shows that we have a vocabulary of 44,276 words. We also can see a sample of the top 100 most used words in the movie reviews. The next step would be to develop a function to save the cleaned text into a file.

# In[20]:


def save_list(lines, filename):
    # convert lines into blob text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    file.write(data)
    file.close


# In[21]:


vocab = Counter()
# add all docs to vocab
process_docs('../input/movie-review/movie_reviews/movie_reviews/pos', vocab) 
process_docs('../input/movie-review/movie_reviews/movie_reviews/neg', vocab) 

min_occurance = 2
tokens = [k for k, c in vocab.items() if c >= min_occurance]
print(len(tokens))

save_list(tokens, 'vocab.txt')


# ## Train CNN with Embedding Layer
# 
# In this section, we will explore the way of word embedding while training a convolutional neural network on the classification problem. 
# 
# A word embedding is a way of representing the text where each word in the vocabulary is represented by a real valued vector in a high dimensional space.
# 
# Words that have similar meaning would have similar presentation in the vector space.
# 
# The real valued vector representation for words can be learned while training the neural network. We can do this in the Keras deep learning library using the `Embedding layer`. 
# 
# The first step is to load the vocabulary. We will use it to filter out words from movie reviews that we are not interested in. We already did that when we save filtered tokens in `vocab.txt` file, we can load it using the `load_doc()`

# In[22]:


vocab_filename = '../input/cleaned-token/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())


# Next, we need to load all of the training data movie reviews. For that we can adapt the process docs() from the previous section to load the documents, clean them, and return them as a list of strings, with one document per string. We want each document to be a string for easy encoding as a sequence of integers later. Cleaning the document involves splitting each review based on white space, removing punctuation, and then filtering out all tokens not in the vocabulary. 

# In[23]:


# turn doc into a clean token
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# In[24]:


# load all docs in a directory
def process_docs(directory, vocab, is_train): 
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'): 
            continue
        if not is_train and not filename.startswith('cv9'): 
            continue
        # create the full path of the file to open
        path = directory + '/' + filename 
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens) 
    return documents


# Combine `neg` and `pos` reviews into one train dataset. 

# In[25]:


# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs('../input/movie-review/movie_reviews/movie_reviews/neg', vocab, is_train)
    pos = process_docs('../input/movie-review/movie_reviews/movie_reviews/pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels


# The next step is to encode each document as a sequence of integers. The Keras Embedding layer requires integer inputs where each integer maps to a single token that has a specific real-valued vector representation within the embedding. These vectors are random at the beginning of training, but during training become meaningful to the network. We can encode the training documents as sequences of integers using the Tokenizer class in the Keras API.

# In[26]:


from keras.preprocessing.text import Tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[27]:


print(Xtrain.shape, Xtest.shape)


# Now that the mapping of words to integers has been prepared, we can use it to encode the reviews in the training dataset. We can do that by calling the texts to sequences() function on the Tokenizer. We also need to ensure that all documents have the same length. This is a requirement of Keras for efficient computation.

# In[28]:


max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)


# In[29]:


import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# load doc into memory
def load_doc(filename):
    # open the file as read only 
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file 
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# load all docs in a directory
def process_docs(directory, vocab, is_train): 
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'): 
            continue
        if not is_train and not filename.startswith('cv9'): 
            continue
            # create the full path of the file to open
        path = directory + '/' + filename # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens) 
    return documents


# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs('../input/movie-review/movie_reviews/movie_reviews/neg', vocab, is_train)
    pos = process_docs('../input/movie-review/movie_reviews/movie_reviews/pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]) 
    return docs, labels


# fit a tokenizer
def create_tokenizer(lines): 
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(lines) 
    return tokenizer


# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs): 
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs) 
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post') 
    return padded


# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length)) 
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# load the vocabulary
vocab_filename = '../input/cleaned-token/vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load training data
train_docs, ytrain = load_clean_dataset(vocab, True) 
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs]) 
print('Maximum length: %d' % max_length)
# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs) 
# define model
model = define_model(vocab_size, max_length)
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# save the model
model.save('model.h5')


# ## Model evaluation
# we will evaluate the trained model and use it to make predictions on new data. First, we can use the built-in evaluate() function to estimate the skill of the model on both the training and test dataset. This requires that we load and encode both the training and test datasets.

# In[30]:


from keras.models import load_model
# load all reviews
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1 
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs]) 
print('Maximum length: %d' % max_length)
# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs) 
Xtest = encode_docs(tokenizer, max_length, test_docs)
# load the model
model = load_model('../input/generated-model/model.h5')
# evaluate model on training dataset
_, acc = model.evaluate(Xtrain, ytrain, verbose=0) 
print('Train Accuracy: %f' % (acc*100))
# evaluate model on test dataset
_, acc = model.evaluate(Xtest, ytest, verbose=0) 
print('Test Accuracy: %f' % (acc*100))


# New data must then be prepared using the same text encoding and encoding schemes as was used on the training dataset. Once prepared, a prediction can be made by calling the predict() function on the model. The function below named predict sentiment() will encode and pad a given movie review text and return a prediction in terms of both the percentage and a label.
# 

# In[31]:


# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model): 
    # clean review
    line = clean_doc(review, vocab)
    # encode and pad review
    padded = encode_docs(tokenizer, max_length, [line])
      # predict sentiment
    yhat = model.predict(padded, verbose=0)
      # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE' 
    return percent_pos, 'POSITIVE'


# In[32]:


text = 'Everyone will enjoy this film. I love it, recommended!'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model) 
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
# test negative text
text = 'This is a bad movie. Do not watch it. It sucks.'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model) 
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))


# Running the example first prints the skill of the model on the training and test dataset. We can see that the model achieves 86.16% accuracy on the training dataset and 70.4% on the test dataset, not a bad score.
# 
# Next, we can see that the model makes the correct prediction on two contrived movie reviews. We can see that the percentage or confidence of the prediction is close to 50% for both, this may be because the two contrived reviews are very short and the model is expecting sequences of 1,000 or more words
