import pandas as pd
import numpy as np
import re
import string
import unicodedata
import nltk

from tqdm import tqdm_notebook
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def remove_punctuation(text):
    import string
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    return text.translate(table)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    new_text = []
    text = text.split()
    for word in text:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_text.append(new_word)
    new_text = ' '.join(new_text)
    return new_text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))  
    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)
    return filtered_sentence 

def clean_text(text):
    text = remove_punctuation(text.lower())
    text = strip_html(text)
    text = remove_non_ascii(text)
    text = remove_stopwords(text)
    result = text.replace(r"\b\w\b","") # remove single char
    return result
