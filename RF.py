from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from nltk import TweetTokenizer
import pandas as pd
import numpy as np
import os
import wandb
import json
import pickle

def glove2bin(glove_model_txt):
    word2vec_file = glove_model_txt.replace('.txt', '_g2w2v.txt')
    bin_file = glove_model_txt.replace('.txt', '_g2w2v.bin')

    if not os.path.exists(bin_file):
        glove2word2vec(glove_input_file=glove_model_txt, word2vec_output_file=word2vec_file)
        glove_model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
        glove_model.save_word2vec_format(bin_file, binary=True)
    else:
        glove_model = KeyedVectors.load_word2vec_format(bin_file, binary=True)
    return glove_model

def getdataset(text, label, model):
    tokenizer = TweetTokenizer()
    avg_ebd_list = []

    for tweet in text:
        tokens = tokenizer.tokenize(tweet)
        ebd_list = []
        for token in tokens:
            if token in model.key_to_index:
                ebd_list.append(model[token])

        if ebd_list:
            avg_ebd = np.mean(np.array(ebd_list), axis=0)
        else:
            avg_ebd = np.zeros(200)
        avg_ebd_list.append(avg_ebd)

    X = np.array(avg_ebd_list)
    Y = np.array(label)

    # shuffle the data
    ids = np.random.permutation(len(X))
    X = X[ids]
    Y = Y[ids]
    return X, Y

def eval_performance(y_true, y_pred, metric_path=None):
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

"""# Data"""

def GetData(fold):
    glove_model = glove2bin('glove.twitter.27B.200d.txt')

    train_df = pd.read_csv(f'TwitterHate_5fold_no_clean/train_TwitterHate_fold{fold}.csv', encoding_errors='ignore')
    train_df.dropna(inplace=True)
    test_df = pd.read_csv(f'TwitterHate_5fold_no_clean/val_TwitterHate_fold{fold}.csv', encoding_errors='ignore')
    test_df.dropna(inplace=True)
    #train_df = pd.read_csv(f'TwitterHate_5fold/train_TwitterHate_fold{self.fold_number}.csv')
    #test_df = pd.read_csv(f'TwitterHate_5fold/val_TwitterHate_fold{self.fold_number}.csv')
    X_train, y_train = getdataset(train_df['tweet'], train_df['label'], glove_model)
    X_test, y_test = getdataset(test_df['tweet'], test_df['label'], glove_model)
    return X_train, y_train, X_test, y_test, test_df

"""# Model"""

if __name__ == "__main__":
    output_folder = 'output_file/RF_no_clean'
    ckpt_folder = 'output_ckpt/RF_no_clean'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    original_output_path = 'output.csv'
    original_metric_path = 'metric.json'

    for fold in range(5):
        print(f"==============================Fold {fold}================================")
        output_path = original_output_path.replace('.csv', f'_fold{fold}.csv')
        output_path = f'{output_folder}/{output_path}'
        metric_path = original_metric_path.replace('.json', f'_fold{fold}.json')
        metric_path = f'{output_folder}/{metric_path}'
        ckpt_path = f'{ckpt_folder}/RF_ckpt_fold{fold}.pkl'

        wandb.init(project = 'MiniProject', name = f'RF_fold{fold}_no_clean')

        # get data
        X_train, y_train, X_test, y_test ,test_df= GetData(fold)

        # get model
        model = RandomForestClassifier()

        # train model
        model.fit(X_train, y_train)

        # save model
        with open(ckpt_path, 'wb') as f:
            pickle.dump(model, f)

        # get pred
        pred = model.predict(X_test).astype(int)

        #save pred
        test_df['pred'] = pred
        test_df.to_csv(output_path, index=False)

        # evaluate
        eval_performance(y_test.tolist(), pred.tolist(), metric_path=metric_path)

        wandb.finish()