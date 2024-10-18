import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler

from transformers import AutoTokenizer
from transformers import BertConfig,BertModel

import json
import os

import wandb
import argparse
import numpy as np
import pandas as pd

from sklearn import metrics
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""#Data"""

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file, encoding_errors='ignore')
        self.data.dropna(inplace=True)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        texts_or_text_pairs = data_row['tweet']
        label = data_row['label']
        features = self.tokenizer.encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = features['input_ids'].squeeze(0)
        token_type_ids = features['token_type_ids'].squeeze(0)
        attention_mask = features['attention_mask'].squeeze(0)

        return input_ids, token_type_ids, attention_mask, label, texts_or_text_pairs

"""#Model"""

class BERTmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased',add_pooling_layer=True, return_dict=False)

        # freeze the parameters in bert encoder except the pooler layer
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.pooler.dense.parameters():  
            param.requires_grad = True

        self.classification = nn.Linear(768, 2)

    def forward(self,
                input_ids: Tensor = None,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None):

        sequence_output, pooled_output = self.bert(input_ids=input_ids, \
                                                    token_type_ids=token_type_ids, \
                                                    attention_mask=attention_mask)
        logits = self.classification(pooled_output)
        logits = nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1)
        return logits, pred

"""#Train"""

class Trainer():
    def __init__(self,
                 fold_number: int = 0,
                 batch_size: int = 16,
                 max_seq_len: int = 256,
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.001,
                 num_epoch: int = 10,
                 output_path: str = 'output.csv',
                 metric_path: str = 'metric.json',
                 *args,
                 **kwargs) -> None:
        self.fold_number = fold_number
        self.model = BERTmodel()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        output_folder = 'output_file/BERT_no_clean'
        ckpt_folder = 'output_ckpt/BERT_no_clean'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        train_file_name = f'TwitterHate_5fold_no_clean/train_TwitterHate_fold{self.fold_number}.csv'
        test_file_name = f'TwitterHate_5fold_no_clean/val_TwitterHate_fold{self.fold_number}.csv'
        self.output_path = output_path.replace('.csv', f'_fold{fold_number}_lr{learning_rate}.csv')
        self.output_path = f'{output_folder}/{self.output_path}'
        self.metric_path = metric_path.replace('.json', f'_fold{fold_number}_lr{learning_rate}.json')
        self.metric_path = f'{output_folder}/{self.metric_path}'
        self.ckpt_path = f'{ckpt_folder}/BERT_linear_ckpt_fold{fold_number}_lr{learning_rate}_best.pth'

        train_dataset = TextDataset(train_file_name,self.tokenizer, max_seq_len)
        test_dataset = TextDataset(test_file_name,self.tokenizer, max_seq_len)

        self.train_dataloader = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=batch_size)

        self.loss_fct = nn.CrossEntropyLoss()
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = AdamW(params = trainable_params,
                            lr = learning_rate,
                            weight_decay = weight_decay)

        self.num_epoch = num_epoch
        self.best_val_loss = float('inf')

    def train(self):

        for epoch in range(self.num_epoch):
            total_train_loss = self.train_step(
                                        epoch = epoch,
                                        num_epochs = self.num_epoch)

            total_test_loss = self.test_step(
                                        epoch = epoch,
                                        num_epochs = self.num_epoch)

            print(f"==============================Epoch {epoch+1}================================")
            print(f'''
                  ### Epoch {epoch+1}:
                  The final train loss is {total_train_loss}
                  The final Test loss is {total_test_loss}''')
            print("==============================================================================")


    def forward_step(self, batch):
        input_ids, token_type_ids, attention_mask, label, texts_or_text_pairs = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        #texts_or_text_pairs = texts_or_text_pairs.to(device)

        logits, pred = self.model.forward(input_ids = input_ids,
                                          token_type_ids = token_type_ids,
                                          attention_mask = attention_mask)
        logits = logits.view(logits.size(0), -1)
        #labels = torch.tensor(labels, dtype = torch.long, device = pred.device).view(-1).detach()
        loss = self.loss_fct(logits, label)

        return pred, loss

    def train_step(self, epoch, num_epochs):
        train_loss_sum = []
        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")

        for train_batch in progress_bar:
            _, train_loss= self.forward_step(train_batch)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            train_loss = train_loss.detach().item()
            progress_bar.set_postfix(train_loss=train_loss)

            wandb.log({'epoch': epoch+1, 'train_loss': train_loss})

            train_loss_sum.append(train_loss)
            torch.cuda.empty_cache()
        return np.mean(train_loss_sum)

    def test_step(self, epoch, num_epochs):
        test_loss_sum = []
        pred_list = []
        label_list = []
        progress_bar = tqdm(self.test_dataloader, desc=f"Test Epoch {epoch+1}/{num_epochs}")
        texts = []
        for test_batch in progress_bar:
            _, _, _, label, texts_or_text_pairs = test_batch
            texts.extend(texts_or_text_pairs)
            label_list.extend(label.tolist())

            pred, test_loss = self.forward_step(test_batch)
            test_loss = test_loss.detach().item()
            test_loss_sum.append(test_loss)
            pred_list.extend([p.item() for p in pred])

        #print('original_pred_list',pred_list)

        df = pd.DataFrame({'label':label_list,'tweet':texts,'pred':pred_list})
        df.to_csv(self.output_path.replace('.csv',f'_epoch{epoch}.csv'), index=False)

        print(f'Evaluate the performance for epoch {epoch}!')
        self.eval_performance(label_list, pred_list, metric_path=self.metric_path.replace('.json',f'_epoch{epoch}.json'))

        test_loss_avg = np.mean(test_loss_sum)
        wandb.log({'test_loss': test_loss_avg})

        # save the first ckpt
        if epoch == 0:
            self.best_ckpt_path = self.ckpt_path.replace('.pth',f'_epoch{epoch}.pth')
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict()}, \
                        self.best_ckpt_path)
            self.best_val_loss = test_loss_avg
        # save the better ckpt
        if test_loss_avg < self.best_val_loss:
            os.remove(self.best_ckpt_path)
            self.best_ckpt_path = self.ckpt_path.replace('.pth',f'_epoch{epoch}.pth')
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict()}, \
                        self.best_ckpt_path)
            self.best_val_loss = test_loss_avg

        torch.cuda.empty_cache()
        return test_loss_avg

    def eval_performance(self, y_true, y_pred, metric_path=None):
        # Precision
        metric_dict = {}
        precision = metrics.precision_score(y_true, y_pred)
        print("Precision:\n\t", precision)
        metric_dict['Precision'] = precision

        # Recall
        recall = metrics.recall_score(y_true, y_pred)
        print("Recall:\n\t",  recall)
        metric_dict['Recall'] = recall

        # Accuracy
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("Accuracy:\n\t", accuracy)
        metric_dict['Accuracy'] = accuracy

        print("-------------------F1, Micro-F1, Macro-F1, Weighted-F1..-------------------------")
        print("-------------------**********************************-------------------------")

        # F1 Score
        f1 = metrics.f1_score(y_true, y_pred)
        print("F1 Score:\n\t", f1)
        metric_dict['F1'] = f1

        # Micro-F1 Score
        micro_f1 =  metrics.f1_score(y_true, y_pred, average='micro')
        print("Micro-F1 Score:\n\t",micro_f1)
        metric_dict['Micro-F1'] = micro_f1

        # Macro-F1 Score
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        print("Macro-F1 Score:\n\t", macro_f1)
        metric_dict['Macro-F1'] = macro_f1

        # Weighted-F1 Score
        weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        print("Weighted-F1 Score:\n\t", weighted_f1)
        metric_dict['Weighted-F1'] = weighted_f1

        print("------------------**********************************-------------------------")
        print("-------------------**********************************-------------------------")

        # ROC AUC Score
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        print("ROC AUC:\n\t", roc_auc)
        metric_dict['ROC-AUC'] = roc_auc

        # Confusion matrix
        confusion_mat = metrics.confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n\t", confusion_mat)
        metric_dict['TN'] = int(confusion_mat[0,0])
        metric_dict['FP'] = int(confusion_mat[0,1])
        metric_dict['FN'] = int(confusion_mat[1,0])
        metric_dict['TP'] = int(confusion_mat[1,1])


        wandb.log({'Precision': precision,
                    'Recall': recall,
                    'Accuracy': accuracy,
                    'F1:': f1,
                    'Micro-F1': micro_f1,
                    'Macro-F1': macro_f1,
                    'Weighted-F1': weighted_f1,
                    'ROC-AUC': roc_auc,
                    'TN': confusion_mat[0,0],
                    'FP': confusion_mat[0,1],
                    'FN': confusion_mat[1,0],
                    'TP': confusion_mat[1,1]})


        if metric_path is not None:
            json.dump(metric_dict,open(metric_path,'w'),indent=4)

"""# Run"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running Tweets Hate speech classification by BERT.')
    parser.add_argument('--fold', metavar='F', type=int, help='fold', default=0)
    parser.add_argument('--lr', metavar='L', type=float, help='lr', default=0.0005)

    args = parser.parse_args()


    fold = args.fold
    lr = args.lr
    #fold = 0
    wandb.init(project = 'MiniProject', name = f'BERT_fold{fold}_lr{lr}_no_clean')

    trainer = Trainer(
                    fold_number = fold,
                    batch_size = 32,
                    max_seq_len = 256,
                    learning_rate = lr,
                    weight_decay = 0.001,
                    num_epoch = 20)

    trainer.train()

    #torch.distributed.destroy_process_group()