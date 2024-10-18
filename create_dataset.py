from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.utils import resample
import os
import re

def get_5_fold_dataset():
    df = pd.read_csv('train_E6oV3lV.csv')
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)

    for fold, ((df_0_train_idx, df_0_val_idx), (df_1_train_idx, df_1_val_idx)) in enumerate(zip(kfold.split(df_0), kfold.split(df_1))):

        df_0_train, df_0_val = df_0.iloc[df_0_train_idx], df_0.iloc[df_0_val_idx]
        df_1_train, df_1_val = df_1.iloc[df_1_train_idx], df_1.iloc[df_1_val_idx]

        val_data = pd.concat([df_0_val, df_1_val])

        #upsampling the training set
        df_1_train_upsampled = resample(df_1_train,
                                        replace=True,
                                        n_samples=len(df_0_train),
                                        random_state=12345)
        train_data = pd.concat([df_0_train, df_1_train_upsampled])

        data_folder = 'TwitterHate_5fold'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        train_data.to_csv(f'{data_folder}/train_TwitterHate_fold{fold}.csv', index=False)
        val_data.to_csv(f'{data_folder}/val_TwitterHate_fold{fold}.csv', index=False)

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[#@&]\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def get_5_fold_dataset_clean_data():
    df = pd.read_csv('train_E6oV3lV.csv')
    # clean the text: remove URL, @XXX, #XXX
    df['tweet'] = df['tweet'].apply(clean_text)

    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)

    for fold, ((df_0_train_idx, df_0_val_idx), (df_1_train_idx, df_1_val_idx)) in enumerate(zip(kfold.split(df_0), kfold.split(df_1))):

        df_0_train, df_0_val = df_0.iloc[df_0_train_idx], df_0.iloc[df_0_val_idx]
        df_1_train, df_1_val = df_1.iloc[df_1_train_idx], df_1.iloc[df_1_val_idx]

        val_data = pd.concat([df_0_val, df_1_val])

        #upsampling the training set
        df_1_train_upsampled = resample(df_1_train,
                                        replace=True,
                                        n_samples=len(df_0_train),
                                        random_state=12345)
        train_data = pd.concat([df_0_train, df_1_train_upsampled])

        data_folder = 'TwitterHate_5fold'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        train_data.to_csv(f'{data_folder}/train_TwitterHate_fold{fold}.csv', index=False)
        val_data.to_csv(f'{data_folder}/val_TwitterHate_fold{fold}.csv', index=False)

get_5_fold_dataset_clean_data()