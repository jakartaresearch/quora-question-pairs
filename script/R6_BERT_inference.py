import pandas as pd
import numpy as np
import argparse
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from LogWatcher import log


device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)


def data_preprocessing(test_data, tokenizer):
    input_ids, token_type_ids, attention_masks = [], [], []

    for sent in test_data.itertuples():
        encoded_dict = tokenizer.encode_plus(
                            text = sent.question1,
                            text_pair = sent.question2,                      
                            add_special_tokens = True,
                            max_length = 64,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                        )

        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(test_data.is_duplicate.values)
    return input_ids, token_type_ids, attention_masks, labels


def data_loader(input_ids, token_type_ids, attention_masks, labels, batch_size):
    pred_data = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=batch_size)
    return pred_data, pred_sampler, pred_dataloader


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='model/')
    parser.add_argument('--test_data', type=str,
                        default='../data/test.tsv')
    parser.add_argument('--batch_size', type=int, default=32)
    opt = parser.parse_args()
    
    logger = log(path="logs/", file="bert_inference.logs")
    
    # Load a trained model and vocabulary that you have fine-tuned
    logger.info('Get BERT pretrained model')
    model = BertForSequenceClassification.from_pretrained(opt.model_path)
    logger.info('Get BERT tokenizer')
    tokenizer = BertTokenizer.from_pretrained(opt.model_path)

    model.to(device)
    logger.info("Load testing data")
    test_data = pd.read_csv(opt.test_data, sep='\t')
    test_data.columns = ['is_duplicate','question1','question2','id']
    
    logger.info("Data Preprocessing")
    input_ids, token_type_ids, attention_masks, labels = data_preprocessing(test_data, tokenizer)
    logger.info("Data Loader")
    pred_data, pred_sampler, pred_dataloader = data_loader(input_ids, token_type_ids, attention_masks, labels, opt.batch_size)
    
    
    logger.info('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
    model.eval()

    predictions , true_labels = [], []
    total_test_loss = 0
    total_test_accuracy = 0
    total_test_f1 = 0
    total_test_prec = 0
    total_test_rec = 0
    
    for batch in pred_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_token_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            loss, logits = model(b_input_ids, token_type_ids=b_token_ids, 
                            attention_mask=b_input_mask, labels=b_labels)
            
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_test_loss += loss.item()
        total_test_accuracy += flat_accuracy(logits, label_ids)
        total_test_f1 += f1_score(label_ids, np.argmax(logits, axis=1))
        total_test_prec += precision_score(label_ids, np.argmax(logits, axis=1))
        total_test_rec += recall_score(label_ids, np.argmax(logits, axis=1))
    
    
    len_data = len(pred_dataloader)
    avg_test_loss = total_test_loss / len_data          
    avg_test_accuracy = total_test_accuracy / len_data
    avg_test_f1 = total_test_f1 / len_data
    avg_test_prec = total_test_prec / len_data
    avg_test_rec = total_test_rec / len_data

    logger.info("Inference testing data is DONE")    
    logger.info("Average loss: {0:.4f}".format(avg_test_loss))
    logger.info("Average accuracy: {0:.4f}".format(avg_test_accuracy*100))
    logger.info("Average f1: {0:.4f}".format(avg_test_f1*100))
    logger.info("Average prec: {0:.4f}".format(avg_test_prec*100))
    logger.info("Average rec: {0:.4f}".format(avg_test_rec*100))