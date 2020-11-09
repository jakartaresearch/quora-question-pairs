import os
import argparse
import pandas as pd
import numpy as np
import torch
import time
import datetime
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from LogWatcher import log
from tqdm import tqdm

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)


class Dataset():
    def __init__(self, data_path, kfold_data_path):
        self.raw_train, self.raw_val = self.load_kfold_data(data_path,
                                                            kfold_data_path)
        self.train, self.val, self.tokenizer = self.data_preprocessing(
            self.raw_train, self.raw_val)

    def remove_row_nan(self, df):
        df = df.dropna(axis=0)
        return df

    def load_kfold_data(self, data_path, kfold_data_path):
        logger.info("Load KFold data")
        data = pd.read_csv(data_path, sep='\t')
        data = self.remove_row_nan(data)
        train_id = pd.read_csv(kfold_data_path + '/train_id.csv')
        val_id = pd.read_csv(kfold_data_path + '/val_id.csv')

        # Get specific data by id
        raw_train = data[data.id.isin(train_id.id.values)]
        raw_val = data[data.id.isin(val_id.id.values)]

        # random sample data
        raw_train = raw_train.sample(frac=1, random_state=42)
        raw_val = raw_val.sample(frac=1, random_state=42)
        return raw_train, raw_val

    def data_preprocessing(self, raw_train, raw_val):
        logger.info("Data Preprocessing")
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        data = [raw_train, raw_val]
        for idx, dt in tqdm(enumerate(data)):
            input_ids, token_type_ids, attention_masks = [], [], []
            for sent in dt.itertuples():
                encoded_dict = tokenizer.encode_plus(
                    text=sent.question1,
                    # Sentence to encode.
                    text_pair=sent.question2,
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=64,           # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    return_attention_mask=True,   # Construct attn. masks.
                    return_tensors='pt',     # Return pytorch tensors.
                )
                input_ids.append(encoded_dict['input_ids'])
                token_type_ids.append(encoded_dict['token_type_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
            # Convert the lists into tensors.
            input_ids = torch.cat(input_ids, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)

            if idx == 0:
                labels = torch.tensor(raw_train.is_duplicate.values)
                train = {'input_ids': input_ids, 'token_type_ids': token_type_ids,
                         'attention_masks': attention_masks, 'labels': labels}
            else:
                labels = torch.tensor(raw_val.is_duplicate.values)
                val = {'input_ids': input_ids, 'token_type_ids': token_type_ids,
                       'attention_masks': attention_masks, 'labels': labels}
        return train, val, tokenizer


def bert_config():
    logger.info("Get BERT pretrained model")
    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-cased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    bert_model.to(device)
    logger.info("Set Adam Optimizer")
    bert_optimizer = AdamW(bert_model.parameters(),
                           lr=2e-5,
                           eps=1e-8
                           )
    return bert_model, bert_optimizer


def create_data_loader(train, val):
    logger.info("Create Data Loader")
    t_input_ids, t_token_type_ids, t_attention_mask, t_labels = train.values()
    train_dataset = TensorDataset(t_input_ids, t_token_type_ids,
                                  t_attention_mask, t_labels)

    train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                   batch_size=opt.batch_size)

    v_input_ids, v_token_type_ids, v_attention_mask, v_labels = val.values()
    val_dataset = TensorDataset(v_input_ids, v_token_type_ids,
                                v_attention_mask, v_labels)
    val_data_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset),
                                 batch_size=opt.batch_size)
    return train_data_loader, val_data_loader

def training(train_dataloader, ):
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    total_train_f1 = 0
    total_train_prec = 0
    total_train_rec = 0

    model.train()
    logger.info("Training...")
    for step, batch in enumerate(train_dataloader):
        if step % 1000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            logger.info('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(
                step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_token_ids = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        model.zero_grad()
        loss, logits = model(b_input_ids,
                             token_type_ids=b_token_ids,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        total_train_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_train_accuracy += flat_accuracy(logits, label_ids)
        total_train_f1 += f1_score(label_ids, np.argmax(logits, axis=1))
        total_train_prec += precision_score(label_ids,
                                            np.argmax(logits, axis=1))
        total_train_rec += recall_score(label_ids,
                                        np.argmax(logits, axis=1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # Calculate the average loss and accuracy over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    avg_train_f1 = total_train_f1 / len(train_dataloader)
    avg_train_prec = total_train_prec / len(train_dataloader)
    avg_train_rec = total_train_rec / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    logger.info("Average loss: {0:.2f}".format(avg_train_loss))
    logger.info("Average accuracy: {0:.2f}".format(avg_train_accuracy))
    logger.info("Average f1: {0:.2f}".format(avg_train_f1))
    logger.info("Average prec: {0:.2f}".format(avg_train_prec))
    logger.info("Average rec: {0:.2f}".format(avg_train_rec))
    logger.info("Training epoch took: {:}".format(training_time))
    return avg_train_loss, avg_train_accuracy, avg_train_f1, avg_train_prec, avg_train_rec, training_time


def validation(val_dataloader):
    logger.info("Validation...")
    t0 = time.time()
    model.eval()

    total_eval_loss = 0
    total_eval_accuracy = 0
    total_eval_f1 = 0
    total_eval_prec = 0
    total_eval_rec = 0

    # Evaluate data for one epoch
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_token_ids = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=b_token_ids,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        total_eval_f1 += f1_score(label_ids, np.argmax(logits, axis=1))
        total_eval_prec += precision_score(label_ids,
                                           np.argmax(logits, axis=1))
        total_eval_rec += recall_score(label_ids,
                                       np.argmax(logits, axis=1))

    # Report the final metrics
    avg_val_loss = total_eval_loss / len(val_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_f1 = total_eval_f1 / len(val_dataloader)
    avg_val_prec = total_eval_prec / len(val_dataloader)
    avg_val_rec = total_eval_rec / len(val_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    logger.info("Average loss: {0:.2f}".format(avg_val_loss))
    logger.info("Average accuracy: {0:.2f}".format(avg_val_accuracy))
    logger.info("Average f1: {0:.2f}".format(avg_val_f1))
    logger.info("Average prec: {0:.2f}".format(avg_val_prec))
    logger.info("Average rec: {0:.2f}".format(avg_val_rec))
    logger.info("Validation took: {:}".format(validation_time))
    return avg_val_loss, avg_val_accuracy, avg_val_f1, avg_val_prec, avg_val_rec, validation_time
    

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='../data/quora_duplicate_questions.tsv')
    parser.add_argument('--kfold_data_path', type=str,
                        default='../data/cross_validation_data/1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    opt = parser.parse_args()
    
    logger = log(path="logs/", file="bert_train.logs")

    quora_dataset = Dataset(opt.dataset_path, opt.kfold_data_path)
    model, optimizer = bert_config()
    train_dataloader, val_dataloader = create_data_loader(quora_dataset.train,
                                                          quora_dataset.val)
    
    total_steps = len(train_dataloader) * opt.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=total_steps)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, opt.epochs):
        logger.info('==== Epoch {:} / {:} ===='.format(epoch_i + 1, opt.epochs))
        
        train_loss, train_accu, train_f1, train_prec, train_rec, training_time = training(train_dataloader)
        val_loss, val_accu, val_f1, val_prec, val_rec, validation_time = validation(val_dataloader)
        
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'KFold': 1,
                'Epoch': epoch_i + 1,
                'Training Loss': train_loss,
                'Validation Loss': val_loss,
                'Training Accuracy': train_accu,
                'Validation Accuracy': val_accu,
                'Training F1': train_f1,
                'Validation F1': val_f1,
                'Training Precision': train_prec,
                'Validation Precision': val_prec,
                'Training Recall': train_rec,
                'Validation Recall': val_rec,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    logger.info("Training complete!")
    logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('Epoch')
    df_stats.to_csv('training_stats.csv')
    
    #Save model
    output_dir = 'model/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir) 
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    quora_dataset.tokenizer.save_pretrained(output_dir)