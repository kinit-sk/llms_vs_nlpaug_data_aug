import pandas as pd
import re
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn import tree
import numpy as np
import itertools
import pickle

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset

from datasets import Dataset, DatasetDict

import argparse
from transformers import enable_full_determinism

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from transformers import RobertaForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import logging
import argparse
import evaluate
import datasets
import random
import os
import shutil
from transformers import AdamW
import tqdm

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Run adversarial training with pre-generated LLM data.')
parser.add_argument('--no_epochs', type=int, const=10, default=10, nargs='?',
                    help='No. normal epochs before running adversarial')
parser.add_argument('--seed', type=int, const=42, default=42, nargs='?',
                    help='Seed to be used for shuffling.')
parser.add_argument('--batch_size', type=int, const=32, default=32, nargs='?',
                    help='Traing batch size.')
parser.add_argument('--batch_size_eval', type=int, const=256, default=256, nargs='?',
                    help='Eval batch size.')
parser.add_argument('--model_save_dir', type=str,
                    help='Where to save the final model to.')
parser.add_argument('--repeat', type=int, const=10, default=10, nargs='?',
                    help='How many times should the process be repeated.')
parser.add_argument('--base_model_type', type=str,
                    help='Specify what kind of mode to use.')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.base_model_type)
random.seed(args.seed)

def prepare_data(df):
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = tokenizer(dataset["text"], padding=True, return_tensors='pt', truncation=True, max_length=128)
    tokenized_datasets['label'] = dataset['label']
    tokenized_datasets['text'] = dataset['text']
    ds = Dataset.from_dict(tokenized_datasets).with_format("torch")
    return ds

def prepare_data_llm(df):
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = tokenizer(dataset["text"], padding=True, return_tensors='pt', truncation=True)
    tokenized_datasets['label'] = dataset['label']
    tokenized_datasets['text'] = dataset['text']
    tokenized_datasets['orig_text'] = dataset['seed']
    ds = Dataset.from_dict(tokenized_datasets).with_format("torch")
    return ds

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def eval_loop(model, ds, eval_batch_size):
    model.eval().to(device)
    test_loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False)
    all_preds = []
    all_corrs = []
    all_txts = []
    
    total_batches = len(test_loader)
    #pbar = tqdm.tqdm(total=total_batches)
    with torch.no_grad():
        total_correct = 0
        for idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            txts = list(batch['text'])

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            _, predicted = torch.max(outputs[1], 1)
            all_preds.extend(predicted)
            all_corrs.extend(labels)
            all_txts.extend(txts)
            
            correct = (predicted == labels).sum().item()
            total_correct+=correct
            #pbar.update(1)
            
    #pbar.close()
    return all_preds, all_corrs

def train_loop(model, ds, BATCH_SIZE=16, NUM_EPOCHS=1):
    train_loader = DataLoader(ds['train'], batch_size=BATCH_SIZE, shuffle=True)
    #test_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.to(device)

    optim = AdamW(model.parameters(), lr=2e-5)
    total_batches = len(train_loader)
    #pbar = tqdm.tqdm(total=total_batches)
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            total_loss += loss.item()
            #pbar.update(1)
        
        logging.info("LOSS: " + str(total_loss/len(train_loader)))
        #pbar.close()        
    return model
    

def get_samples_per_seed(df_orig, df_aug_0, df_aug_1, no_seeds, no_samples):
    labels = list(df_orig['label'].unique()) 

    df_orig_sub = df_orig.groupby('label', group_keys=False).apply(lambda x: x.sample(no_seeds, random_state=args.seed))
    df_orig_sub = df_orig_sub.rename(columns={'text': 'seed'})
    
    df_orig_sub['seed'] = df_orig_sub['seed'].astype(str)
    df_aug_0['seed'] = df_aug_0['seed'].astype(str)
    df_aug_1['seed'] = df_aug_1['seed'].astype(str)
    
    df_data = df_orig_sub.merge(df_aug_0, how='inner', on=['seed']).rename(columns={'label_x':'label'})[['text', 'label', 'seed']]
    dfs = [v for k, v in df_data.groupby('seed')]
    
    total_dfs = []
    for df_sub in dfs:
        try:
            total_dfs.append(df_sub.groupby('seed', group_keys=False).apply(lambda x: x.sample(no_samples, random_state=args.seed)))
        except:
            total_dfs.append(df_sub)
            
    df_data = df_orig_sub.merge(df_aug_1, how='inner', on=['seed']).rename(columns={'label_x':'label'})[['text', 'label', 'seed']]
    dfs = [v for k, v in df_data.groupby('seed')]
    for df_sub in dfs:
        try:
            total_dfs.append(df_sub.groupby('seed', group_keys=False).apply(lambda x: x.sample(no_samples, random_state=args.seed)))
        except:
            total_dfs.append(df_sub)
    
    tmp_data = pd.concat(total_dfs)
    return tmp_data.sample(frac=1, random_state=args.seed).reset_index(drop=True).drop_duplicates().dropna()[['text', 'label']], df_orig_sub.rename(columns={'seed': 'text'}).dropna()

df_orig = pd.read_csv('sst5/data/sst_seeds_train.csv').sample(frac=1, random_state=args.seed).reset_index(drop=True).drop_duplicates().dropna()
df_orig['text'] = df_orig['text'].str.lower()
#df_orig['label'] = (~df_orig.label.astype(bool)).astype(int)
ds_train_orig = prepare_data(df_orig)
df_test_orig = pd.read_csv('sst5/data/test_sst.csv').drop_duplicates().dropna()
#df_test_orig['label'] = (~df_test_orig.label.astype(bool)).astype(int)
ds_test_orig = prepare_data(df_test_orig)

ds_dct_orig = datasets.DatasetDict({"train":ds_train_orig,"test":ds_test_orig})

from transformers import AutoModelForSequenceClassification, TextClassificationPipeline
from sklearn import metrics

methods = ['cont_ins', 'back']
lst_no_seeds_samples = [5,10,20,30,40,50,100]
lst_no_coll_samples = [1, 2, 5, 10, 15]

dct_seeds_to_inc = {42: '', 1: '_1', 2: '_2'}

dct_model_to_dir = {'FacebookAI/roberta-base': 'results_roberta', 'google-bert/bert-base-uncased':'results_bert', 'distilbert/distilbert-base-uncased':'results_distilbert'}

#del old_model
torch.cuda.empty_cache()

for no_seed_samples in lst_no_seeds_samples:
    logging.info('Loading LLM data for training')
    train_file0 = 'sst5/data/sst_'+methods[0]+'.csv'
    train_file1 = 'sst5/data/sst_'+methods[1]+'.csv'
    df_aug_full_0 = pd.read_csv(train_file0)
    df_aug_full_1 = pd.read_csv(train_file1)
    for no_coll_samples in lst_no_coll_samples:
        res_orig = []

        for repeat_no in range(0, args.repeat):
            logging.info('***************** Now on repeat {} **************.'.format(repeat_no))
            logging.info('Running normal training for {} epochs.'.format(args.no_epochs))
            
            if args.base_model_type == 'distilbert/distilbert-base-uncased':
                model = AutoModelForSequenceClassification.from_pretrained(args.base_model_type, num_labels=5, dropout= 0.2)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(args.base_model_type, num_labels=5, classifier_dropout= 0.2)
            
            df_aug, df_orig_sub = get_samples_per_seed(df_orig, df_aug_full_0, df_aug_full_1, no_seed_samples, no_coll_samples)
            dfs = [df_orig_sub, df_aug]
            
            df_train = pd.concat(dfs).sample(frac=1, random_state=args.seed).reset_index(drop=True).drop_duplicates()
            df_train['text'] = df_train['text'].str.lower()
            ds_train = prepare_data(df_train)
            
            ds_dct_train = datasets.DatasetDict({"train":ds_train,"test":ds_test_orig})
            
            if no_seed_samples < 20:
                batch_size = 16
            elif no_seed_samples >= 20 and no_seed_samples < 40:
                batch_size = 32
            else:
                batch_size = 64
            
            model = train_loop(model, ds_dct_train, BATCH_SIZE=batch_size, NUM_EPOCHS=args.no_epochs)
                
            logging.info('Running evaluation on orig data.')
            
            preds, corrs = eval_loop(model, ds_test_orig, args.batch_size_eval)
            preds = [x.item() for x in preds]
            corrs = [x.item() for x in corrs]
            res = metrics.accuracy_score(corrs, preds)
            res_orig.append(res)
            logging.info('ACC orig: {}'.format(res))
            torch.cuda.empty_cache()

        del model
        dct_df = {'res_orig': res_orig}
        df_res = pd.DataFrame.from_dict(dct_df)
        res_file = 'sst5/data/'+dct_model_to_dir[args.base_model_type]+''+dct_seeds_to_inc[args.seed]+'/sst_comb_seeds_'+str(no_seed_samples)+'_coll_'+str(no_coll_samples)+'_class.csv'
        df_res.to_csv(res_file, index=False)