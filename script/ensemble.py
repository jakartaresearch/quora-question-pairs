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
from joblib import dump
from cleansing import clean_text
from LogWatcher import log


def read_csv(path):
    """Read file csv.

    Args:
        path (str): path file

    Returns:
        dataFrame: dataFrame of content file with tab separator
    """
    d_data = pd.read_csv(path, sep='\t')

    return d_data


def remove_row_nan(df):
    """Remove missing value per row.

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
    """Run a simple cleansing to remove uncessary text.

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
    """Transform text into vector.

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
    """Concatenate question1 and question2.

    Args:
        q1 (array): question1
        q2 (array): question2

    Returns:
        array: return an array concatenation of q1 and q2
    """
    return scipy.sparse.hstack((q1, q2))


def metrics(y_true, y_pred):
    """Calculate metrics.

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
    """Run experiment of a given algorithm.

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
    """Model wrapper to run all experiment ensemble models.

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


def fit_model_all(d_train, d_test, algo, extranction):
    """Fitting all the data to the model 

    Args:
        X (array): data input
        y (array): data target
        algo (model): class algorithm

    Returns:
        model: return a model after fit
    """
    d_data = pd.concat((d_train, d_test), axis=0)
    d_data.reset_index(drop=True, inplace=True)

    cv1 = extranction()
    q1 = cv1.fit_transform(d_data.q1_clean)
    cv2 = extranction()
    q2 = cv2.fit_transform(d_data.q2_clean)

    X = concat(q1, q2)
    y = d_data.is_duplicate.values
    model = algo()
    model.fit(X, y)

    return cv1, cv2, model


def wrapper_fit_all(d_train, d_test, model_path):
    cv1, cv2, model = fit_model_all(
        d_train, d_test, XGBClassifier, CountVectorizer)
    save_model([cv1, cv2, model], os.path.join(model_path, "cv_xgb.pkl"))

    cv1, cv2, model = fit_model_all(
        d_train, d_test, CatBoostClassifier, CountVectorizer)
    save_model([cv1, cv2, model], os.path.join(model_path, "cv_cat.pkl"))

    tfidf1, tfidf2, model = fit_model_all(
        d_train, d_test, XGBClassifier, TfidfVectorizer)
    save_model([tfidf1, tfidf2, model], os.path.join(
        model_path, "tfidf_xgb.pkl"))

    tfidf1, tfidf2, model = fit_model_all(
        d_train, d_test, CatBoostClassifier, TfidfVectorizer)
    save_model([tfidf1, tfidf2, model], os.path.join(
        model_path, "tfidf_cat.pkl"))


def save_model(model, filename):
    """[summary]

    Args:
        model ([type]): [description]
        filename ([type]): [description]
    """
    dump(model, filename)


def main(split_folders, test_scenario, report_path, model_path):
    """Run all process.

    Args:
        split_folders (str): cross val folders
        test_scenario (str): parameter for testing code
        report_path (str): path for report directory
        model_path (str): path for saved model
    """

    split_folders = glob.glob(os.path.join(split_folders, "*"))
    score_list = []
    for split_index, path in enumerate(split_folders, 1):
        test_path, train_path = glob.glob(os.path.join(path, '*'))
        logger.info("step 1/7 :read data")
        d_train, d_test = read_csv(train_path), read_csv(test_path)

        logger.info("step 2/7 :remove nan")
        d_train, d_test = remove_row_nan(d_train), remove_row_nan(d_test)

        d_train = d_train.sample(frac=1)
        d_test = d_test.sample(frac=1)

        d_train.reset_index(inplace=True)
        d_test.reset_index(inplace=True)

        if test_scenario:
            logger.warning(
                "running using test scenario, only collect 10 samples")
            d_train = d_train.loc[:10, :]
            d_test = d_test.loc[:10, :]

        # cleansing step
        logger.info("step 3/7 :cleansing")
        d_train["q1_clean"] = d_train.question1.apply(clean_text)
        d_train["q2_clean"] = d_train.question2.apply(clean_text)
        d_test["q1_clean"] = d_test.question1.apply(clean_text)
        d_test["q2_clean"] = d_test.question2.apply(clean_text)

        y_train = d_train.is_duplicate.values
        y_test = d_test.is_duplicate.values

        # transformation step
        logger.info("step 4/7 :transforming cv")
        cv_q1, x_train_q1, x_test_q1 = transform(
            d_train.q1_clean, d_test.q1_clean, CountVectorizer)
        cv_q2, x_train_q2, x_test_q2 = transform(
            d_train.q2_clean, d_test.q2_clean, CountVectorizer)

        x_train = concat(x_train_q1, x_train_q2)
        x_test = concat(x_test_q1, x_test_q2)

        # modeling
        logger.info("step 5/7 :fitting")
        feat = "count vectorizer"
        scores = model_wrapper(x_train, x_test, y_train,
                               y_test, split_index, feat)
        score_list.extend(scores)

        logger.info("step 6/7 :transforming tfidf")
        cv_q1, x_train_q1, x_test_q1 = transform(
            d_train.q1_clean, d_test.q1_clean, TfidfVectorizer)
        cv_q2, x_train_q2, x_test_q2 = transform(
            d_train.q2_clean, d_test.q2_clean, TfidfVectorizer)

        x_train = concat(x_train_q1, x_train_q2)
        x_test = concat(x_test_q1, x_test_q2)

        logger.info("step 7/7 :fitting")
        feat = "tfidf"
        scores = model_wrapper(x_train, x_test, y_train,
                               y_test, split_index, feat)
        score_list.extend(scores)

    logger.info("saving reports...")
    today = datetime.today().strftime("%d-%m-%Y")
    pickle.dump(score_list, open(
        f'{report_path}/{today}_reports.pkl', 'wb'))

    # fit all data to the model
    wrapper_fit_all(d_train, d_test, model_path)
    logger.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-path", help="path for data train", default="data/cross_validation_data", required=True)
    parser.add_argument(
        "-t", "--test-scenario", help="testing for code running", action="store_true")
    parser.add_argument(
        "-r", "--report-path",
        help="path for saving report model performance", default="reports", required=True)
    parser.add_argument(
        "--model-path", help="model path for vectorizer and model", default="models", required=True)
    args = parser.parse_args()
    logger = log(path="logs/", file="bert_inference.logs")
    main(args.data_path, args.test_scenario, args.report_path, args.model_path)
