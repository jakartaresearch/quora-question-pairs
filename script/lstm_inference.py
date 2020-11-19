import os
import argparse
from tokenizers import BertWordPieceTokenizer
import torch
from lstm_train import QuoraClassifier
from utils import decode_label
from LogWatcher import log


def load_tokenizer(path):
    tokenizer = BertWordPieceTokenizer(os.path.join(path, 'vocab.txt'))
    return tokenizer


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

    return model


def setup_model(num_vocab, emb_size, hid_size, num_class):
    model = QuoraClassifier(num_vocab, emb_size, hid_size, num_class)
    return model


def encode_input(tokenizer, q1, q2):
    x_raw = tokenizer.encode(args.q1, args.q2)
    x = x_raw.ids
    x = torch.LongTensor([x])

    return x


def main(args):
    logger.info("start load tokenizer")
    tokenizer = load_tokenizer(args.tokenizer_path)
    num_vocab = tokenizer.get_vocab_size()
    logger.info("build model")
    model = setup_model(num_vocab, args.emb_size,
                        args.hid_size, args.num_class)
    logger.info("load model")
    model = load_model(model, args.model_path)
    x = encode_input(tokenizer, args.q1, args.q2)
    logger.info("predict the label")
    y_pred = model(x)
    y_pred = y_pred.argmax().unsqueeze(dim=0)
    print(decode_label(y_pred.detach().numpy()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="model path", required=True)
    parser.add_argument("--tokenizer-path",
                        help="tokenizer path saved from train", required=True)
    parser.add_argument("--q1", help="question 1", required=True)
    parser.add_argument("--q2", help="question 2", required=True)
    parser.add_argument(
        "--emb-size", help="embedding size for embedding layer", default=512, type=int)
    parser.add_argument(
        "--hid-size", help="hidden size in lstm", default=512, type=int)
    parser.add_argument(
        "--num-class", help="number of class target", default=2, type=int)
    args = parser.parse_args()
    logger = log(path="logs/", file="lstm.log")
    main(args)
