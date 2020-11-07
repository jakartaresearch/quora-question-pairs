"""Benchmark Boosting Model (XGBoost and CatBoost)."""
import argparse
import os
import glob
import pickle
import string
import scipy
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from cleansing import clean_text


def read_csv(path):
    """Read file csv

    Args:
        path (str): path file

    Returns:
        dataFrame: dataFrame of content file with tab separator
    """
    d_data = pd.read_csv(path, sep='\t')

    return d_data


def remove_row_nan(df):
    """Remove missing value per row

    Args:
        df (dataFrame): dataFrame

    Returns:
        dataFrame: dataFrame with no missing value
    """
    df = df.dropna(axis=0)
    return df


def remove_punctuation(text):
    """Remove punctuation in text.

    Args:
        text (str): an input of text

    Returns:
        str: text after remove the puctuation
    """
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def simple_cleansing(text):
    """[summary]

    Args:
        text (str): an input of text

    Returns:
        str: return a clean text
    """
    text = text.lower()
    text = remove_punctuation(text)
    stopword = stopwords.words('english')
    word_list = text.split()
    word_clean = [word for word in word_list if word not in stopword]
    text = " ".join(word_clean)

    return text


def transform(train, test, vectorizer):
    """[summary]

    Args:
        train (series): a series of train text data
        test (series): a series of test text data
        vectorizer (function): a vectorizer function, CountVectorizer/TfidfVectorizer

    Returns:
        list: return a list of vec, train_feature, test feature
    """
    vec = vectorizer()
    train_feat = vec.fit_transform(train)
    test_feat = vec.transform(test)

    return (vec, train_feat, test_feat)


def concat(q1, q2):
    """Concatenate question1 and question2

    Args:
        q1 (array): question1
        q2 (array): question2

    Returns:
        array: return an array concatenation of q1 and q2
    """
    return scipy.sparse.hstack((q1, q2))


def metrics(y_true, y_pred):
    """[summary]

    Args:
        y_true (array): actual label
        y_pred (array): prediction label

    Returns:
        dict: return dictionary contain accuracy, precision, recall, and f score.
    """
    accu = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    reca = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {"accuracy": accu, "precision": prec, "recall": reca, "f1_score": f1}


def model(x_train, x_test, y_train, y_test, algo):
    """Run experiment of a given algorithm

    Args:
        x_train (array): input train
        x_test (array): input test
        y_train (array): label train
        y_test (array): label test
        algo (Class): Class algorithm from scikit learn

    Returns:
        dict: return score dictionary
    """
    algo.fit(x_train, y_train)
    y_pred = algo.predict(x_test)

    score_dict = metrics(y_test, y_pred)

    return score_dict


def model_wrapper(x_train, x_test, y_train, y_test, split_index, feat):
    """A model wrapper to run all experiment ensemble models.

    Args:
        x_train ([type]): [description]
        x_test ([type]): [description]
        y_train ([type]): [description]
        y_test ([type]): [description]
        split_index ([type]): [description]
        feat ([type]): [description]

    Returns:
        [type]: [description]
    """
    score_list = []
    print("\tXGBoost")
    score_dict = model(x_train, x_test, y_train, y_test, XGBClassifier())
    score_dict["split_index"] = split_index
    score_dict["model"] = "xgboost"
    score_dict["feature"] = feat
    score_list.append(score_dict)

    print("\tCatboost")
    score_dict = model(x_train, x_test, y_train, y_test,
                       CatBoostClassifier(verbose=False))
    score_dict["split_index"] = split_index
    score_dict["model"] = "catboost"
    score_dict["feature"] = feat
    score_list.append(score_dict)

    return score_list


def main(split_folders, test_scenario, report_path):

    split_folders = glob.glob('../data/cross_validation_data/*')
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

        if test_scenario:
            d_train = d_train.loc[:10, :]
            d_test = d_test.loc[:10, :]

        # cleansing step
        print("step 3/7 :cleansing...")
        d_train["q1_clean"] = d_train.question1.apply(clean_text)
        d_train["q2_clean"] = d_train.question2.apply(clean_text)
        d_test["q1_clean"] = d_test.question1.apply(clean_text)
        d_test["q2_clean"] = d_test.question2.apply(clean_text)

        y_train = d_train.is_duplicate.values
        y_test = d_test.is_duplicate.values

        # transformation step
        print("step 4/7 :transforming cv...")
        cv_q1, x_train_q1, x_test_q1 = transform(
            d_train.q1_clean, d_test.q1_clean, CountVectorizer)
        cv_q2, x_train_q2, x_test_q2 = transform(
            d_train.q2_clean, d_test.q2_clean, CountVectorizer)

        x_train = concat(x_train_q1, x_train_q2)
        x_test = concat(x_test_q1, x_test_q2)

        # modeling
        print("step 5/7 :fitting...")
        feat = "count vectorizer"
        scores = model_wrapper(x_train, x_test, y_train,
                               y_test, split_index, feat)
        score_list.extend(scores)

        print("step 6/7 :transforming tfidf...")
        cv_q1, x_train_q1, x_test_q1 = transform(
            d_train.q1_clean, d_test.q1_clean, TfidfVectorizer)
        cv_q2, x_train_q2, x_test_q2 = transform(
            d_train.q2_clean, d_test.q2_clean, TfidfVectorizer)

        x_train = concat(x_train_q1, x_train_q2)
        x_test = concat(x_test_q1, x_test_q2)

        print("step 7/7 :fitting...")
        feat = "tfidf"
        scores = model_wrapper(x_train, x_test, y_train,
                               y_test, split_index, feat)
        score_list.extend(scores)

        print("done...")

    today = datetime.today().strftime("%d-%m-%Y")
    pickle.dump(score_list, open(
        f'{report_path}/{today}_reports.pkl', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", help="path for data train", required=True)
    parser.add_argument(
        "test-scenario", help="testing for code running", default=False, required=True)
    parser.add_argument(
        "--report_path", help="path for saving report model performance", default="reports", required=True)
    parser.add_argument(
        "--model_path", "model path for vectorizer and model", default="models", required=True)

    main()
