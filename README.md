# Quora Question Pairs

Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers. --Kaggle

Start Project: 25 April 2020
End Project: 31 May 2020

Download data [here](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)

Download Clean Data [here](https://drive.google.com/open?id=1_y-K7YJsLg9uIivTsFY_I9uh93FACOlF)

Download Cross Validation Data [here](https://drive.google.com/open?id=18haftEPePeBsVv3dlIPkkpeO29a49ERL)

## Branch

    |--master
    |--dev

## Main References

| Title                                                                                    | Author           | Year |
| ---------------------------------------------------------------------------------------- | ---------------- | ---- |
| Aiming beyond the obvious: Identifying non-obvious cases in semantic similarity datasets | Peinelt et al.   | 2020 |
| Bilateral multi-perspective matching for natural language sentences                      | Wang et al.      | 2017 |
| Large Scale Question Paraphrase Retrieval with Smoothed Deep Metric Learning             | Bonadiman et al. | 2019 |
| Retrofitting Contextualized Word Embeddings with Paraphrases                             | Shi et al.       | 2019 |

## Requirements

```
pip install -r requirements.txt
```

## Result

| Name  | Stack             | Score |
| ----- | ----------------- | ----- |
| Exp-1 | CV - XGBoost      | 68.09 |
| Exp-2 | CV - Catboost     | 74.66 |
| Exp-3 | TF-IDF - XGBoost  | 69.14 |
| Exp-4 | TF-IDF - Catboost | 75.39 |

We also experimented using deep learning

| Name            | Stack | Score |
| --------------- | ----- | ----- |
| Basic LSTM      | LSTM  | 80.32 |
| BERT-Base-Cased | BERT  | 97.07 |

## Quick Start

- You need to download all the data needed from [here](https://drive.google.com/drive/folders/1LpZhav8bftTSqsZHIQEdI_JWT4S-797L?usp=sharing)

To use the Benchmark on Ensemble Algorithms
Train

```
python script/ensemble_train.py --data-path data/cross_validation_data --report-path reports --model-path models
```

Inference

```
python script/ensemble_inference.py --model-path models/cv_cat.pkl --q1 "where are you going" --q2 "where will you go"
```

To use the LSTM Model
Train

```
python script/lstm_train.py --cross-path data/cross_validation_data --tokenizer-path data/bert-case --batch-size 50
```

Inference

```
python script/lstm_inference.py --model-path models/bi_lstm.pth --tokenizer-path data/bert-case --q1 "where are you going" --q2 "where will you go"
```

To use the BERT Model
Train

```
python script/BERT_train.py --dataset_path data/quora_duplicate_questions.tsv --kfold_data_path data/cross_validation_data/1 --model_path model 
```

Inference

```
python script/BERT_inference.py --model_path model --q1 "where are you going" --q2 "where will you go"
```

## Contributors

[![](https://github.com/andreaschandra/git-assets/blob/master/pictures/andreas.png)](https://github.com/andreaschandra)
[![](https://github.com/andreaschandra/git-assets/blob/master/pictures/ruben.png)](https://github.com/rubentea16)
