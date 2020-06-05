# Quora Question Pairs
Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers. --Kaggle

Start Project: 25 April 2020

Download Clean Data [here](https://drive.google.com/open?id=1_y-K7YJsLg9uIivTsFY_I9uh93FACOlF)

Download Cross Validation Data [here](https://drive.google.com/open?id=18haftEPePeBsVv3dlIPkkpeO29a49ERL)

## Branch

    |--master
    |--dev
    
## Main References
Title|Author|Year
---|---|---
Aiming beyond the obvious: Identifying non-obvious cases in semantic similarity datasets|Peinelt et al.|2020
Bilateral multi-perspective matching for natural language sentences|Wang et al.|2017
Large Scale Question Paraphrase Retrieval with Smoothed Deep Metric Learning|Bonadiman et al.|2019
Retrofitting Contextualized Word Embeddings with Paraphrases|Shi et al.|2019

## Requirements
```
pip install -r requirements.txt
```

## Result
Name|Stack|Score
---|---|---
Exp-1|CV - XGBoost|68.09
Exp-2|CV - Catboost|74.66
Exp-3|TF-IDF - XGBoost|69.14
Exp-4|TF-IDF - Catboost|75.39

## Quick Start
1. You need download the dataset first in cross-validation
2. all scripts are located in notebook directory
3. <A|R><NUM> means A was written by Andreas and R by Ruben. <NUM> means the order to run the experiments
 
