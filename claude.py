import anthropic
import pandas as pd
import json
import wandb
import re
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn import metrics

class Create_example():
    def __init__(self, example_df, k):
        self.k = k
        #self.example_df = example_df
        self.example_df_label_1 = example_df[example_df['label'] == 1]
        self.example_df_label_0 = example_df[example_df['label'] == 0]
            
    def forward(self, text_label):
        if text_label == 1:
            example_list = self.example_df_label_1.sample(n=self.k)['example'].tolist()
        elif text_label == 0:
            example_list = self.example_df_label_0.sample(n=self.k)['example'].tolist()

        output_text = '\n\n### Example:\n'.join(example_list)
        return output_text
    
def generate_IO_prompt(data_point):
    if data_point:
        return f"""
        ### Instruction:
        You are a Twitter hate speech classification classifier. Assign a correct label of the Input text from ['Not Hateful', 'Hateful']. Only return the label without any other texts.

        ### Input:
        {data_point}

        ### Response:

        """
    
def generate_IO_prompt_kshot(data_point, example_text):
    if data_point:
        return f"""
### Example:
{example_text}

————————————————————————————————————————

        ### Instruction:
        You are a Twitter hate speech classification classifier. Assign a correct label of the Input text from ['Not Hateful', 'Hateful']. Only return the label without any other texts.

        ### Input:
        {data_point}

        ### Response:

        """

def generate_CoC_prompt(data_point):
    if data_point:
        return f"""
        ### Instruction:
        You are a Twitter hate speech classification classifier. Assign a correct label of the Input text from ['Not Hateful', 'Hateful'].

        ### Input:
        {data_point}

        You can choose to output the result directly if you believe your judgment is reliable,
        or
        You think step by step if your confidence in your judgment is less than 90%:
        Step 1: What is the SURFACE sentiment, as indicated by clues such as keywords, sentimental phrases, emojis?
        Step 2: Deduce what the sentence really means, namely the TRUE intention, by carefully checking any rhetorical devices, language style, etc.
        Step 3: Compare and analysis Step 1 and Step 2, infer the final hate label.

        ### Response:

        ### Label: 

        """

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running Tweets Hate speech classification by Claude.')
    parser.add_argument('--fold', metavar='F', type=int, help='fold', default=0)
    parser.add_argument('--chunks', metavar='C', type=int, help='number of chunks', default=3)
    parser.add_argument('--strategy', metavar='S', type=str, help='strategy', default='IO')
    parser.add_argument('--api_key', metavar='A', type=str, help='api key', default='')
    parser.add_argument('--k_shot', metavar='K', type=int, help='k shot', default=1)
    parser.add_argument('--use_kshot', metavar='U', type=bool, help='use kshot', default=False)

    args = parser.parse_args()

    fold = args.fold
    api_key = args.api_key
    chunks = args.chunks
    strategy = args.strategy
    k = args.k_shot
    use_kshot = args.use_kshot
    
    # log
    if not use_kshot:
        wandb.init(project = 'MiniProject', name = f'Claude_{strategy}_fold{fold}_no_clean_new')
    elif use_kshot:
        wandb.init(project = 'MiniProject', name = f'Claude_{strategy}_{k}shot_fold{fold}_no_clean')

    # data
    output_folder = f'output_file/Claude_no_clean/{strategy}'
    ckpt_folder = f'output_ckpt/Claude_no_clean/{strategy}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    output_path = f'{output_folder}/output_fold{fold}_new.csv'
    metric_path = f'{output_folder}/metric_fold{fold}_new.json'

    df = pd.read_csv(f'TwitterHate_5fold_no_clean/val_TwitterHate_fold{fold}.csv', encoding_errors='ignore')
    #df = pd.read_csv(f'test_OLID_no_clean.csv', encoding_errors='ignore')
    #df = pd.read_csv(f'TwitterHate_5fold/try.csv', encoding_errors='ignore')
    df.dropna(inplace=True)

    if use_kshot:
        example_df = pd.read_csv(f'Claude_examples_5fold_no_clean/claude_examples_fold{fold}.csv', encoding_errors='ignore')
        example_df.dropna(inplace=True)

        example_creator = Create_example(example_df,k)

        output_folder = f'output_file/Claude_{k}shot_no_clean/{strategy}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        output_path = f'{output_folder}/output_{k}shot_fold{fold}.csv'
        metric_path = f'{output_folder}/metric_{k}shot_fold{fold}.json'

    # load api
    client = anthropic.Anthropic(api_key=api_key)

    chunk_size = int(np.ceil(len(df) / chunks))
    df_chunks = []

    for chunk_num in range(chunks):
        chunk_file_path = output_path.replace('.csv',f'_chunk{chunk_num}.csv')

        if os.path.exists(chunk_file_path):
            df_chunk = pd.read_csv(chunk_file_path)
            df_chunks.append(df_chunk)
            continue

        df_chunk = df[chunk_num*chunk_size:min(len(df), (chunk_num+1)*chunk_size)]
        output_texts = []
        prompts_list = []
        labels = []

        for index, ( _, row) in enumerate(tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Processing chunk {chunk_num + 1}/{chunks}")):
            if use_kshot:
                example_text = example_creator.forward(int(row['label']))
                if strategy == 'IO':
                    content = generate_IO_prompt_kshot(row['tweet'], example_text)
            else:
                if strategy == 'IO':
                    content = generate_IO_prompt(row['tweet'])
                elif strategy == 'COC':
                    content = generate_CoC_prompt(row['tweet'])
            
            message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=512,
                    messages=[
                        {"role": "user", "content": content}
                    ]
                )
            prompts_list.append(content)
            result = message.content[0].text
            result = result.lower().strip()
            output_texts.append(result)

            if re.search(r"\bnot hateful\b", result, re.IGNORECASE):
                labels.append(0)
            else:
                labels.append(1)

        df_chunk['llm_prompt'] = prompts_list
        df_chunk['llm_output'] = output_texts
        df_chunk['pred']= labels
        df_chunk.to_csv(chunk_file_path, index=0)
        df_chunks.append(df_chunk)

    df = pd.concat(df_chunks)

    df.to_csv(output_path, index=0)
    for i in range(chunks):
        chunk_file_path = output_path.replace('.csv',f'_chunk{i}.csv')
        if os.path.exists(chunk_file_path):
            os.remove(chunk_file_path)
    eval_performance(df['label'], df['pred'], metric_path)