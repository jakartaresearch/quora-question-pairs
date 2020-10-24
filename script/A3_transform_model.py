#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import pickle
import string
from datetime import datetime
from tqdm import tqdm, notebook

from cleansing import clean_text

import pandas as pd
import numpy as np
import scipy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[ ]:


tqdm.pandas()


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


split_folders = glob.glob('../data/cross_validation_data/*')


# In[ ]:


def read_csv(path):
    d_data = pd.read_csv(path, sep='\t')
    
    return d_data


# In[ ]:


def remove_row_nan(df):
    df = df.dropna(axis = 0)
    return df


# In[ ]:


table = str.maketrans('', '', string.punctuation)

def remove_punctuation(text):
    return text.translate(table)


# In[ ]:


def simple_cleansing(text):
    text = text.lower()
    text = remove_punctuation(text)
    stopword = stopwords.words('english')
    word_list = text.split()
    word_clean = [word for word in word_list if word not in stopword]
    text = " ".join(word_clean)
    
    return text


# In[ ]:


def transform(train, test, vectorizer):
    vec = vectorizer()
    train_feat = vec.fit_transform(train)
    test_feat = vec.transform(test)
    
    return (vec, train_feat, test_feat)


# In[ ]:


def concat(q1, q2):
    return scipy.sparse.hstack((q1, q2))


# In[ ]:


def metrics(y_true, y_pred):
    accu = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    reca = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {"accuracy": accu, "precision": prec, "recall": reca, "f1_score": f1}


# In[ ]:


def model(x_train, x_test, y_train, y_test, algo):
    algo.fit(x_train, y_train)
    y_pred = algo.predict(x_test)
    
    score_dict = metrics(y_test, y_pred)
    
    return score_dict


# In[ ]:


def model_wrapper(x_train, x_test, y_train, y_test, split_index, feat):
    score_list = []
    print("\tXGBoost")
    score_dict = model(x_train, x_test, y_train, y_test, XGBClassifier())
    score_dict["split_index"] = split_index
    score_dict["model"] = "xgboost"
    score_dict["feature"] = feat
    score_list.append(score_dict)
    
    print("\tCatboost")
    score_dict = model(x_train, x_test, y_train, y_test, CatBoostClassifier(verbose=False))
    score_dict["split_index"] = split_index
    score_dict["model"] = "catboost"
    score_dict["feature"] = feat
    score_list.append(score_dict)
    
    return score_list


# In[ ]:


test = False


# In[ ]:


score_list = []
for split_index, path in enumerate(split_folders, 1):
    test_path, train_path = glob.glob(os.path.join(path, '*'))
    print("step 1/7 :read data")
    d_train, d_test = read_csv(train_path), read_csv(test_path)
    print("step 2/7 :remove nan")
    d_train, d_test = remove_row_nan(d_train), remove_row_nan(d_test)
    d_train = d_train.sample(frac=1)
    d_test = d_test.sample(frac=1)
    d_train.reset_index(inplace=True)
    d_test.reset_index(inplace=True)
    
    if test:
        d_train = d_train.loc[:99, :]
        d_test = d_test.loc[:99, :]

    ## cleansing step
    print("step 3/7 :cleansing...")
    d_train["q1_clean"] = d_train.question1.apply(clean_text)
    d_train["q2_clean"] = d_train.question2.apply(clean_text)
    d_test["q1_clean"] = d_test.question1.apply(clean_text)
    d_test["q2_clean"] = d_test.question2.apply(clean_text)
    
    y_train = d_train.is_duplicate.values
    y_test = d_test.is_duplicate.values
    
    ## transformation step
    print("step 4/7 :transforming cv...")
    cv_q1, x_train_q1, x_test_q1 = transform(d_train.q1_clean, d_test.q1_clean, CountVectorizer)
    cv_q2, x_train_q2, x_test_q2 = transform(d_train.q2_clean, d_test.q2_clean, CountVectorizer)
    
    x_train = concat(x_train_q1, x_train_q2)
    x_test = concat(x_test_q1, x_test_q2)
    
    ## modeling
    print("step 5/7 :fitting...")
    feat = "count vectorizer"
    scores = model_wrapper(x_train, x_test, y_train, y_test, split_index, feat)
    score_list.extend(scores)
    
    print("step 6/7 :transforming tfidf...")
    cv_q1, x_train_q1, x_test_q1 = transform(d_train.q1_clean, d_test.q1_clean, TfidfVectorizer)
    cv_q2, x_train_q2, x_test_q2 = transform(d_train.q2_clean, d_test.q2_clean, TfidfVectorizer)
    
    x_train = concat(x_train_q1, x_train_q2)
    x_test = concat(x_test_q1, x_test_q2)
    
    print("step 7/7 :fitting...")
    feat = "tfidf"
    scores = model_wrapper(x_train, x_test, y_train, y_test, split_index, feat)
    score_list.extend(scores)
    
    print("done...")


# In[ ]:


today = datetime.today().strftime('%d-%m-%Y')
pickle.dump(score_list, open('../reports/{}_reports.pkl'.format(today), 'wb'))


# In[ ]:




