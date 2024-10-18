from transformers import BertConfig,BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch 
from typing import Optional
from collections import defaultdict
from datetime import datetime
from lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from src.models.qml_bert_module import QMLBertLitModule
import datasets
import math
import numpy as np
import pandas as pd
from sklearn import metrics
import json
        
class TransformerLitModule(LightningModule):

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        task_name: str,
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        encoder_trainable: float = False,
        projection_trainable: float = True,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        n_unitary_circuit_layers: int = 20,
        text_save_dir: str = 'output_file/QCBERT/',
        fold_num: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.task_name = task_name
        num_labels = 2
        self.num_labels = num_labels
        self.save_hyperparameters()

        self.text_save_dir = text_save_dir
        self.fold_num = fold_num

        self.config = BertConfig.from_json_file(config_path)

        self.num_qubits = self.config.n_qubits
        self.num_unitary_circuit_layers = n_unitary_circuit_layers

        self.model = QMLBertLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                                           bert_config=config_path, 
                                                           quantum_config=config_path,
                                                           pretrained_encoder=False)
        
        # Freeze pre-trained encoder when asked
        for param in self.model.bert.parameters():
            param.requires_grad = True
        if not encoder_trainable:
            for param in self.model.bert.parameters():
                param.requires_grad = False
                
        self.entangle_params_transform_layer = self.model.cls.entangle_params_transform_layer

        
        self.unitary_params = torch.nn.Parameter(torch.Tensor(self.num_unitary_circuit_layers, self.num_qubits, 3))
        self.model.cls.unitary_params.requires_grad = False
        for param in self.model.cls.predictions.parameters():
            param.requires_grad = False 
        for param in self.model.cls.out.parameters():
            param.requires_grad = False 
        for name, param in self.model.named_parameters():
            print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
        
        
        # Freeze classical-to-quantum projection net when asked
        if not projection_trainable:
            for param in self.entangle_params_transform_layer.parameters():
                param.requires_grad = False


        self.num_qubits = self.model.cls.num_qubits
        self.output_layer = nn.Linear(2**self.num_qubits, self.num_labels)
        self.outputs = defaultdict(list)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, texts_or_text_pairs=None):
        
        sequence_output = self.model.bert(input_ids=input_ids, \
                                              token_type_ids=token_type_ids, \
                                              attention_mask=attention_mask)[:1][0]
        
        entangle_params = self.entangle_params_transform_layer(sequence_output[:, 0])
        entangle_params = torch.sigmoid(entangle_params.reshape(-1,self.model.cls.num_entangle_circuit_layers,self.model.cls.num_qubits))* 2 * math.pi
        
        qcircuit_output = self.model.cls.quantum_layer(entangle_params, self.unitary_params)
        
        output_logits = self.output_layer(qcircuit_output.float())
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(output_logits.view(-1, self.num_labels), labels.view(-1))

        output = (output_logits,)
        
        if loss is not None:
            return ((loss,) + output),texts_or_text_pairs
        else:
            return output,texts_or_text_pairs

    def training_step(self, batch, batch_idx):
        outputs, _ = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs, texts_or_text_pairs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]

        self.outputs[dataloader_idx].append({"loss": val_loss, "preds": preds, "labels": labels, 'tweet':texts_or_text_pairs})

    def on_validation_epoch_end(self):
        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        preds = torch.cat([x["preds"] for x in flat_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in flat_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        preds_list = preds.tolist()  
        labels_list = labels.tolist()  

        text = []
        for x in flat_outputs:
            for item in x["tweet"]:
                text.append(item)
        df = pd.DataFrame(columns=['tweet', 'label','pred'])
        for pred, label, text_or_text_pair in zip(preds, labels, text):
            df = pd.concat([df, pd.DataFrame({'tweet': [text_or_text_pair], 'label': [label], 'pred':[pred]})], ignore_index=True)
            
        df.to_csv(f'{self.text_save_dir}/output_{self.task_name}_fold{self.fold_num}_epoch{self.current_epoch}.csv', index=False)
        
        if self.trainer.sanity_checking:
            self.outputs.clear()
            return
        self.eval_performance(labels_list, preds_list, metric_path=f'{self.text_save_dir}/metric_{self.task_name}_fold{self.fold_num}_epoch{self.current_epoch}.json')
        self.outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs, texts_or_text_pairs = self(**batch)
        test_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        
        self.outputs[dataloader_idx].append({"loss": test_loss, "preds": preds, "labels": labels, 'tweet':texts_or_text_pairs})

    def on_test_epoch_end(self):
        
        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        preds = torch.cat([x["preds"] for x in flat_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in flat_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
        
        self.log("test_loss", loss, prog_bar=True)

        preds_list = preds.tolist()  
        labels_list = labels.tolist()  

        text = []
        for x in flat_outputs:
            for item in x["tweet"]:
                text.append(item)
        df = pd.DataFrame(columns=['tweet', 'label','pred'])
        for pred, label, text_or_text_pair in zip(preds, labels, text):
            df = pd.concat([df, pd.DataFrame({'tweet': [text_or_text_pair], 'label': [label], 'pred':[pred]})], ignore_index=True)
            
        df.to_csv(f'{self.text_save_dir}/output_{self.task_name}_fold{self.fold_num}.csv', index=False)
        self.eval_performance(labels_list, preds_list, metric_path=f'{self.text_save_dir}/metric_{self.task_name}_fold{self.fold_num}.json')


        self.outputs.clear()


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer_grouped_parameters = [
            {
                "params":[p for n, p in self.entangle_params_transform_layer.named_parameters()] +\
                    [p for n, p in self.output_layer.named_parameters()] +\
                    [self.unitary_params],  
                "weight_decay": self.hparams.weight_decay,
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
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

        self.log('Precision',precision)
        self.log('Recall',recall)
        self.log('Accuracy',accuracy)
        self.log('F1',f1)
        self.log('Micro-F1',micro_f1)
        self.log('Macro-F1',macro_f1)
        self.log('Weighted-F1',weighted_f1)
        self.log('ROC-AUC',roc_auc)
        self.log('TN',confusion_mat[0,0])
        self.log('FP',confusion_mat[0,1])
        self.log('FN',confusion_mat[1,0])
        self.log('TP',confusion_mat[1,1])



        if metric_path is not None:
            json.dump(metric_dict,open(metric_path,'w'),indent=4)