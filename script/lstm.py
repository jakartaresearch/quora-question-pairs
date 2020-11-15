"""Long Short-term Memory for Quora Question Pairs."""
import os
import glob
import time
import pickle
import argparse
import pandas as pd
from tokenizers import BertWordPieceTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from LogWatcher import log


class DatasetPairs(Dataset):
    """Dataset Class."""

    def __init__(self, cross_val_paths, model_path):
        self.dataset = self.read_dataset(cross_val_paths)
        self.split_dict = self.get_id_cross_val(cross_val_paths)
        self.tokenizer = self.get_tokenizer(model_path)
        self.splited_data(k=1)
        self.set_split(split='train')

    def read_dataset(self, path):
        files = glob.glob(os.path.join(path, '1', '*'))
        df_list = []
        for file in files:
            data = self.read_csv(file)
            df_list.append(data)

        d_data = pd.concat(df_list, axis=0)
        d_data.reset_index(drop=True, inplace=True)

        return d_data

    def read_csv(self, path):
        d_data = pd.read_csv(path, sep='\t')
        return d_data

    def get_id_cross_val(self, path):
        data_dict = {}
        paths = glob.glob(os.path.join(path, '*', '*'))

        if os.name == 'nt':
            paths = [path.replace("\\", "/") for path in paths]

        path_dict = dict((file.split('/')[-2], file) for file in paths)
        for k, path in path_dict.items():
            train = self.read_csv(path)
            id_train = train.id.tolist()

            path = path.replace('train.csv', 'test.csv')
            test = self.read_csv(path)
            id_test = test.id.tolist()

            data_dict[int(k)] = (id_train, id_test)

        return data_dict

    def get_tokenizer(self, path):
        tokenizer = BertWordPieceTokenizer(os.path.join(path, 'vocab.txt'))
        return tokenizer

    def splited_data(self, k):
        id_train, id_test = self.split_dict[k]
        train = self.dataset[self.dataset.id.isin(id_train)]
        test = self.dataset[self.dataset.id.isin(id_test)]

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        self.data_dict = {'train': (train, len(
            train)), 'test': (test, len(test))}

    def set_split(self, split='train'):
        self.data, self.length = self.data_dict[split]

    def __getitem__(self, idx):
        q1 = self.data.loc[idx, "question1"]
        q2 = self.data.loc[idx, "question2"]

        x_raw = self.tokenizer.encode(q1, q2)
        x = x_raw.ids
        y = self.data.loc[idx, "is_duplicate"]

        x = torch.LongTensor(x)
        y = torch.LongTensor([y])

        return (x, y)

    def __len__(self):
        return self.length


class QuoraClassifier(nn.Module):
    def __init__(self, num_vocab, emb_size, hid_size, num_class):
        super(QuoraClassifier, self).__init__()
        self.emb = nn.Embedding(num_vocab, emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
        self.fc = nn.Linear(hid_size, num_class)

    def forward(self, input_):
        out = self.emb(input_)
        out, (h, c) = self.lstm(out)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


def compute_accuracy(y, y_pred):
    y_label = y_pred.argmax(dim=1)
    n_correct = torch.eq(y, y_label).sum().item()
    accuracy = (n_correct / len(y_label)) * 100

    return accuracy


def compute_time(start, end):
    duration = end - start
    m = int(duration / 60)
    s = int(duration % 60)

    return m, s


def padding(data):
    x_list = []
    y_list = []
    for x, y in data:
        x_list.append(x)
        y_list.append(y)

    x_pad = pad_sequence(x_list, batch_first=True)
    y_pad = pad_sequence(y_list, batch_first=True)

    return x_pad, y_pad


def save_report(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def main(args):
    logger.info("Start running...")

    MODEL_PATH = os.path.join(args.model_path, "bi_lstm.pth")
    REPORT_PATH = os.path.join(args.report_path, "bi_lstm.pkl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Running on {device}")

    dataset = DatasetPairs(args.cross_path, args.tokenizer_path)
    logger.info("dataset class created successfully")

    num_vocab = dataset.tokenizer.get_vocab_size()
    logger.debug(f"number of vocab {num_vocab}")

    model = QuoraClassifier(num_vocab, args.emb_size,
                            args.hid_size, args.num_class)
    logger.info("model class created successfully")
    model = model.to(device)

    parameters = sum(p.numel() for p in model.parameters())
    logger.info(f'model has {parameters:,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    logger.debug(f"learning rate set to {args.lr}")

    history = {"running_loss": [], "running_loss_v": [],
               "running_accu": [], "running_accu_v": []}

    logger.info("start training...")
    for epoch in range(1, 101):

        running_loss = 0
        running_loss_v = 0
        running_accu = 0
        running_accu_v = 0

        start = time.time()

        dataset.set_split("train")
        data_gen = DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=padding)
        model.train()
        for batch_index, (x, y) in enumerate(data_gen, 1):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.squeeze().to(device)

            out = model(x)

            loss = criterion(out, y)
            loss_ = loss.item()
            running_loss += (loss_ - running_loss) / batch_index

            accuracy = compute_accuracy(y, out)
            running_accu += (accuracy-running_accu) / batch_index

            loss.backward()
            optimizer.step()

        dataset.set_split("test")
        data_gen = DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=padding)
        model.eval()
        for batch_index, (x, y) in enumerate(data_gen, 1):
            x = x.to(device)
            y = y.squeeze().to(device)

            out = model(x)

            loss = criterion(out, y)
            loss_ = loss.item()
            running_loss_v += (loss_ - running_loss_v) / batch_index

            accuracy = compute_accuracy(y, out)
            running_accu_v += (accuracy - running_accu_v) / batch_index

        end = time.time()
        m, s = compute_time(start, end)

        logger.info(f"{epoch} | {m}m {s}s")
        logger.info(
            f'\ttrain loss: {running_loss:.2f} | train accuracy: {running_accu:.2f}')
        logger.info(
            f'\tval loss: {running_loss_v:.2f} | val accuracy: {running_accu_v:.2f}')

    history["running_loss"].append(running_loss)
    history["running_loss_v"].append(running_loss_v)
    history["running_accu"].append(running_accu)
    history["running_accu_v"].append(running_accu_v)

    logger.debug(f"save to {REPORT_PATH}")
    save_report(history, os.path.join(REPORT_PATH))
    logger.info("save report done")
    logger.debug(f"save to {MODEL_PATH}")
    save_model(model, MODEL_PATH)
    logger.info("save mdoel done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-path", help="cross validation data path", required=True)
    parser.add_argument(
        "--tokenizer-path", help="tokenizer path using tokenizers", required=True)
    parser.add_argument(
        "--emb-size", help="embedding size for embedding layer", default=512, type=int)
    parser.add_argument(
        "--hid-size", help="hidden size in lstm", default=512, type=int)
    parser.add_argument(
        "--num-class", help="number of class target", default=2, type=int)
    parser.add_argument(
        "--batch-size", help="number of batch size", default=256, type=int)
    parser.add_argument(
        "--lr", help="learning rate for adam optimizer", default=0.001, type=int)
    parser.add_argument(
        "--report-path", help="path for train test loss and accuracy to save", default="reports")
    parser.add_argument(
        "--model-path", help="model path", default="models")
    args = parser.parse_args()

    logger = log(path="logs/", file="lstm.log")
    main(args)
