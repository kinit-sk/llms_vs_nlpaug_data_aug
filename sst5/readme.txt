This folder contains results, scripts and data used for the scripts for dataset SST-5.

The scripts folder contains: 
clean_train file for training using only the seed samples for various models
train_peft_lora_*.py scripts which are used for training QLoRA for various models and data
full_training_*.py scripts which are used for training full models for various models and data

The data folder contains:
Datasets collected from ChatGPT (sst_gpt_{method}.csv)
Seed data used (sst_seeeds_*.csv)
NLPAug data collected (sst_{nlpaug_method}.csv)
LLama3 data collected (sst_paraphrases_train_{method}_w_tryout.csv)

The results folder contains:
Roberta, Bert and Distilbert results for full finetuning and LoRA finetuning using LLama3 and NLPAug data (csv files without the prefix 'results')
Roberta, Bert and Distilbert results for full finetuning and LoRA finetuning using GPT-3.5 data (csv files with the prefix 'results')
