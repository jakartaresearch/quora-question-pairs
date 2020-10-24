#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pandas as pd
from nltk.tokenize import word_tokenize

from tqdm import tqdm


# In[ ]:


tqdm.pandas()


# In[ ]:


glob.glob("../data/cross_validation_data/1/*")


# In[ ]:


print("reading csv")


# In[ ]:


d_train = pd.read_csv("../data/cross_validation_data/1/train.csv", sep="\t")
d_test = pd.read_csv("../data/cross_validation_data/1/test.csv", sep="\t")


# In[ ]:


def remove_row_nan(df):
    df = df.dropna(axis = 0)
    return df


# In[ ]:


print("removing nan")


# In[ ]:


d_train, d_test = remove_row_nan(d_train), remove_row_nan(d_test)


# In[ ]:


print("lower string")


# In[ ]:


d_train["question1"]=d_train.question1.str.lower()
d_train["question2"]=d_train.question2.str.lower()
d_test["question1"]=d_test.question1.str.lower()
d_test["question2"]=d_test.question2.str.lower()


# In[ ]:


print("tokenizing word")


# In[ ]:


d_train["token_q1"] = d_train.question1.apply(word_tokenize)
d_train["token_q2"] = d_train.question2.apply(word_tokenize)


# In[ ]:


print("creating corpus")


# In[ ]:


corpus = d_train.token_q1.to_list() + d_train.token_q2.to_list()


# In[ ]:


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# In[ ]:


print("length corpus:", len(corpus))


# In[ ]:


print("creating tagged document")


# In[ ]:


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]


# In[ ]:


print("building doc2vec")


# In[ ]:


model = Doc2Vec(documents, vector_size=128, window=2, min_count=10, workers=4)


# In[ ]:


print("save model")


# In[ ]:


model.save("docvec")


# In[ ]:


print("done")

