import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
import pickle
from datasets import load_dataset
import pandas as pd
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import numpy as np


from transformers.utils import logging

logging.set_verbosity(40) # only log errors

import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CURL_CA_BUNDLE'] = ''

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    tokenizer=tokenizer
)
   
    
import re
import string

def collect_samples(dct_final_prompts):
    dct_responses = {}

    for idx, key in enumerate(dct_final_prompts):
        logging.info('Now on class index: {}'.format(idx))
        dct_responses[key] = []
        for count, prompt in enumerate(dct_final_prompts[key]):
            
            messages = [
                {"role": "system", "content": "You are a helpful data augmentator."},
                {"role": "user", "content": prompt[0]},
            ]
            
            prompt = pipeline.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )
            
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = pipeline(
                prompt,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=1.0
            )
            
            if count > 0 and count % 50 == 0:
                logging.info('Now on count: {}'.format(count))
            print(outputs[0])
            dct_responses[key].append((outputs[0], prompt[1]))
            
    return dct_responses

    
for phase in ['train']:
    get_subsampled = pd.read_csv('yelp/data/yelp_seeds_'+phase+'_1.csv')
        
    dct_phrases = {}
    for key in set(get_subsampled['label']):
        dct_phrases[key] = list(get_subsampled[get_subsampled['label'] == key]['text'])

    default_prompt = 'Please provide 15 different changes of the Text by paraphrasing it. Output the full sentences. Output in format "1. sentence 1, 2. sentence 2, ... , 15. sentence 15". Text: "{}".'

    dct_final_prompts = {}

    for key in dct_phrases:
        dct_final_prompts[key] = []
        for phrase in dct_phrases[key]:
            dct_final_prompts[key].append((default_prompt.format(phrase), phrase))

    logging.info("Starting to collect samples")
            
    dct_responses = collect_samples(dct_final_prompts)
    
    with open('yelp/data/yelp_paraphrases_'+phase+'_para_w_tryout_1.pkl', 'wb') as handle:
        pickle.dump(dct_responses, handle)

for phase in ['train']:
    get_subsampled = pd.read_csv('yelp/data/yelp_seeds_'+phase+'_1.csv')
        
    dct_phrases = {}
    for key in set(get_subsampled['label']):
        dct_phrases[key] = list(get_subsampled[get_subsampled['label'] == key]['text'])

    default_prompt = 'Please provide 15 different changes of the Text by inserting words into the Text. Output the full sentences. Output in format "1. sentence 1, 2. sentence 2, ... , 15. sentence 15". Text: "{}".'

    dct_final_prompts = {}

    for key in dct_phrases:
        dct_final_prompts[key] = []
        for phrase in dct_phrases[key]:
            dct_final_prompts[key].append((default_prompt.format(phrase), phrase))

    logging.info("Starting to collect samples")
            
    dct_responses = collect_samples(dct_final_prompts)
    
    with open('yelp/data/yelp_paraphrases_'+phase+'_ins_w_tryout_1.pkl', 'wb') as handle:
        pickle.dump(dct_responses, handle)
        
for phase in ['train']:
    get_subsampled = pd.read_csv('yelp/data/yelp_seeds_'+phase+'_1.csv')
        
    dct_phrases = {}
    for key in set(get_subsampled['label']):
        dct_phrases[key] = list(get_subsampled[get_subsampled['label'] == key]['text'])

    default_prompt = 'Please provide 15 different changes of the Text by swapping words for their synonyms. Output the full sentences. Output in format "1. sentence 1, 2. sentence 2, ... , 15. sentence 15". Text: "{}".'

    dct_final_prompts = {}

    for key in dct_phrases:
        dct_final_prompts[key] = []
        for phrase in dct_phrases[key]:
            dct_final_prompts[key].append((default_prompt.format(phrase), phrase))

    logging.info("Starting to collect samples")
            
    dct_responses = collect_samples(dct_final_prompts)
    
    with open('yelp/data/yelp_paraphrases_'+phase+'_swap_w_tryout_1.pkl', 'wb') as handle:
        pickle.dump(dct_responses, handle)
