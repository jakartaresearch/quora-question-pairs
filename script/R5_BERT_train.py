import argparse
import pandas as pd
import numpy as np
import torch
import time
import datetime

from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup


device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_type)


class Dataset():
    def __init__(self, data_path, kfold_data_path):
        self.raw_train, self.raw_val = self.load_kfold_data(data_path,
                                                            kfold_data_path)
        self.train, self.val = self.data_preprocessing(
            self.raw_train, self.raw_val)

    def remove_row_nan(self, df):
        df = df.dropna(axis=0)
        return df

    def load_kfold_data(self, data_path, kfold_data_path):
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
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        data = [raw_train, raw_val]
        for idx, dt in enumerate(data):
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
        return train, val


def bert_config():
    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-cased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    bert_model.to(device)
    bert_optimizer = AdamW(model.parameters(),
                           lr=2e-5,
                           eps=1e-8
                           )
    return bert_model, bert_optimizer


def create_data_loader(train, val):
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
        print('==== Epoch {:} / {:} ===='.format(epoch_i + 1, opt.epochs))
        t0 = time.time()
        total_train_loss = 0
        total_train_accuracy = 0
        total_train_f1 = 0
        total_train_prec = 0
        total_train_rec = 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 1000 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(
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

        print("  Average loss: {0:.2f}".format(avg_train_loss))
        print("  Average accuracy: {0:.2f}".format(avg_train_accuracy))
        print("  Average f1: {0:.2f}".format(avg_train_f1))
        print("  Average prec: {0:.2f}".format(avg_train_prec))
        print("  Average rec: {0:.2f}".format(avg_train_rec))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Validation...")
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

        print("  Average loss: {0:.2f}".format(avg_val_loss))
        print("  Average accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Average f1: {0:.2f}".format(avg_val_f1))
        print("  Average prec: {0:.2f}".format(avg_val_prec))
        print("  Average rec: {0:.2f}".format(avg_val_rec))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'KFold': 1,
                'Epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Training Accuracy': avg_train_accuracy,
                'Validation Accuracy': avg_val_accuracy,
                'Training F1': avg_train_f1,
                'Validation F1': avg_val_f1,
                'Training Precision': avg_train_prec,
                'Validation Precision': avg_val_prec,
                'Training Recall': avg_train_rec,
                'Validation Recall': avg_val_rec,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time()-total_t0)))
