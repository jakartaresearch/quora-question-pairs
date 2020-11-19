"""Ensemble Inference."""
import os
import argparse
from joblib import load
import numpy as np
from utils import decode_label
from ensemble_train import clean_text, concat
from LogWatcher import log


def load_model(model_path):
    if os.path.exists(model_path):
        model = load(model_path)
        return model
    else:
        ValueError("model not found")


def main(model_path, q1, q2):
    logger.info("load model")
    vectorizer_1, vectorizer_2, model = load_model(model_path)
    logger.info("text cleansing")
    q1 = clean_text(q1)
    q2 = clean_text(q2)
    logger.info("text transformation")
    vec_q1 = vectorizer_1.transform(np.array([q1]))
    vec_q2 = vectorizer_2.transform(np.array([q2]))
    questions = concat(vec_q1, vec_q2)
    logger.info("predict the label")
    y_pred = model.predict(questions)
    print(decode_label(y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="model path", required=True)
    parser.add_argument("--q1", help="question 1", required=True)
    parser.add_argument("--q2", help="question 2", required=True)
    args = parser.parse_args()
    logger = log(path="logs/", file="ensemble.log")
    main(args.model_path, args.q1, args.q2)
