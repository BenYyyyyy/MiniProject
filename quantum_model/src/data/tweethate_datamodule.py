from transformers import BertModel, AutoTokenizer
from transformers import BertConfig
import datasets
from lightning import LightningDataModule
from ast import literal_eval
import pandas as pd
# conver the dataset into pytorch_lighting datamudule
from torch.utils.data import DataLoader
class tweethate_DataModule(LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
        "texts_or_text_pairs",
      ]
    def __init__(
        self,
        model_name_or_path: str = 'qbert_config',
        dataset_path: str = 'TwitterHate_5fold_no_clean/',
        task_name: str = 'sst2',
        fold_num: int = 0,
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        test_batch_size: int = 32,
        **kwargs):

        super().__init__()
        self.dataset_path = dataset_path
        self.fold_num = fold_num
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size

        self.num_labels = 2

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
                )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        #self.eval_splits_val = [x for x in self.dataset.keys() if "validation" in x]
        #self.eval_splits_test = [x for x in self.dataset.keys() if "test" in x]

  
  
    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        texts_or_text_pairs = example_batch['tweet']

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]
        # added
        features["texts_or_text_pairs"] = texts_or_text_pairs

        return features
  
    def prepare_data(self):
        self.dataset = datasets.DatasetDict()
        data_file_list = ['train', 'val']
        dataset_name_list = ['train', 'validation']
        for index, file_name in enumerate(data_file_list):
            dataframe = pd.read_csv(f'{self.dataset_path}/{file_name}_TwitterHate_fold{self.fold_num}.csv', encoding_errors='ignore')
            dataframe.dropna(inplace=True)

            data = dataframe['tweet'].tolist()
            labels = dataframe['label'].tolist()
            data_dict = {
                        'tweet':data,\
                        'label':labels
                        }
            self.dataset[dataset_name_list[index]]=datasets.Dataset.from_dict(data_dict)
        
        print('len of train set:',len(self.dataset['train']))
        print('len of val set:',len(self.dataset['validation']))
      
            

        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        