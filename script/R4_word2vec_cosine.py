"""Modelling use word2vec and cosine similarity."""
import argparse
import pickle
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from LogWatcher import log
from tqdm import tqdm_notebook, tqdm
tqdm_notebook().pandas()


def remove_row_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing value per row.

    Args:
        df (dataFrame): dataFrame
    Returns:
        dataFrame: dataFrame with no missing value
    """
    df = df.dropna(axis=0)
    return df


def average_weight(w: list, len_q: int) -> np.array:
    """Calculate average weight for sentence.

    Args:
        w: list of weight per word
        len_q: number of word in sentence
    Returns:
        average weight for sentence
    """
    weight = np.sum(w, axis=0)
    weight = np.divide(weight, len_q)
    weight = weight.reshape(1, -1)
    return weight


def question_similarity(row, weights) -> float:
    """Calculate similarity between question1 and question2 using cosine similarity.

    Args:
        row: row data in DataFrame
        weights: word2vec embedding
    Returns:
        sim: cosine similarity between 0 to 1
    """

    q1_weight, q2_weight = [], []

    q1 = row['clean_question1']
    q2 = row['clean_question2']

    len_q1 = len(q1.split())
    len_q2 = len(q2.split())

    for word in str(q1).split():
        q1_weight.append(weights.get(word, np.zeros(300)))
    for word in str(q2).split():
        q2_weight.append(weights.get(word, np.zeros(300)))

    avg_q1_weight = average_weight(q1_weight, len_q1)
    avg_q2_weight = average_weight(q2_weight, len_q2)
    sim = cosine_similarity(avg_q1_weight, avg_q2_weight)
    return sim


def main(clean_data: str, kfold_data: str, word_embed: str):
    data = pd.read_csv(clean_data)
    data = remove_row_nan(data)

    logger.info("Load Word Embedding")
    with open(word_embed, 'rb') as file:
        weights = pickle.load(file)

    total_acc, total_f1, total_prec = [], [], []

    kfold_folder = glob.glob(kfold_data + '/*')
    train_id_file = '/train_id.csv'
    val_id_file = '/val_id.csv'

    for kf, path in enumerate(kfold_folder, 1):
        logger.info('Load KFold data from = %s', path)
        logger.info('KFold -%s', kf)
        train_id = pd.read_csv(path + train_id_file)
        val_id = pd.read_csv(path + val_id_file)

        # Get specific data by id
        train = data[data.id.isin(train_id.id.values)]
        val = data[data.id.isin(val_id.id.values)]

        # random sample data
        train = train.sample(frac=1, random_state=42)
        val = val.sample(frac=1, random_state=42)

        # calculate cosine similarity
        logger.info('calculate cosine similarity')
        x_train = train.progress_apply(
            question_similarity, args=(weights), axis=1)
        x_val = val.progress_apply(question_similarity, args=(weights), axis=1)

        # Prepare train, val data
        x_train = x_train.values.reshape(-1, 1)
        x_val = x_val.values.reshape(-1, 1)
        y_train = train.is_duplicate.values
        y_val = val.is_duplicate.values

        # Classifier
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        # Prediction
        y_pred = clf.predict(x_val)
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)

        logger.info('KFold -%s Accuracy: %s', kf, accuracy)
        logger.info('KFold -%s F1: %s', kf, f1)
        logger.info('KFold -%s Precision: %s', kf, prec)
        total_acc.append(accuracy)
        total_f1.append(f1)
        total_prec.append(prec)

    logger.info('Accuracy: %s', sum(total_acc)/5)
    logger.info('F1: %s', sum(total_f1)/5)
    logger.info('Precision: %s', sum(total_prec)/5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data', type=str,
                        default='')
    parser.add_argument('--kfold_data', type=str,
                        default='../data/cross_validation_data')
    parser.add_argument('--word_embed', type=str,
                        default='../model/w2v_embed.pkl')
    opt = parser.parse_args()

    logger = log(path="logs/", file="word2vec_cosine.logs")
    main(opt.clean_data, opt.kfold_data, opt.word_embed)
