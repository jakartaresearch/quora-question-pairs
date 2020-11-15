"""Create Word Embedding using Word2Vec."""
import argparse
import multiprocessing
import time
import pickle
import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from LogWatcher import log

nltk.download('punkt')


def combine_all_questions(dt: pd.DataFrame) -> list:
    """Combine question1 and question2 data.

    Args:
        dt: dataset
    Returns:
        questions: combined of question1 and question2
    """
    clean_q1 = dt['clean_question1'].values
    clean_q2 = dt['clean_question2'].values

    questions = [str(val) for sublist in [clean_q1, clean_q2]
                 for val in sublist]
    return questions


def tokenizer(questions: list) -> list:
    """Tokenize each question sentence."""
    sentences = [sent_tokenize(str(text)) for text in questions]
    sentences = [sent[0].split() for sent in sentences]
    return sentences


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.start_time = 0
        self.loss_previous_step = 0

    def on_epoch_begin(self):
        print("Epoch #{} start".format(self.epoch))
        self.start_time = time.time()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Time execution per epoch : {}'.format(
            time.time() - self.start_time))

        self.loss_previous_step = loss

        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(
                self.epoch, loss-self.loss_previous_step))

        self.epoch += 1


def main(clean_data):
    """Run all process."""
    cores = multiprocessing.cpu_count()
    data = pd.read_csv(clean_data)
    logger.info("Combine question1 and question2")
    questions = combine_all_questions(data)
    logger.info("Tokenize all sentences")
    sent = tokenizer(questions)

    logger.info("Initialize Word2Vec Class")
    w2v_model = Word2Vec(min_count=2,
                         window=5,
                         size=300,
                         sg=1,  # if 1, then skipgram is used; if 0, then cbow
                         workers=cores)

    logger.info("Build Vocabulary")
    w2v_model.build_vocab(sent)
    logger.info("Train Word2Vec model")
    w2v_model.train(sentences=sent,
                    total_examples=w2v_model.corpus_count,
                    epochs=20,
                    report_delay=1,
                    compute_loss=True,  # set compute_loss = True
                    callbacks=[callback()])  # add the callback class

    logger.info("Save word embedding pickle")
    w2v = dict(zip(w2v_model.wv.index2word, w2v_model.wv.vectors))
    with open("model/w2v_embed.pkl", "wb") as file:
        pickle.dump(w2v, file)

    logger.info("Save Word2Vec model")
    w2v_model.save('model/word2vec.model')

    w2v_model.wv.most_similar(['dog'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data', type=str,
                        default='')
    opt = parser.parse_args()

    logger = log(path="logs/", file="word2vec.log")

    main(opt.clean_data)
