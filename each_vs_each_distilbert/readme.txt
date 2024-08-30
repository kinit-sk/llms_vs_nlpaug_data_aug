This folder contains aggregated results and visualizations for DistilBERT full finetuning

The csv files are aggregated results with comparisons of two methods for each combination of finetuning for no. collected samples, no. seed samples per label and random seeds.

Folder 'hist' contains a histogram of times when one of the compared methods worked better.
Folder 'clean_viz' contains a histogram of times when one of the compared methods worked better or worse against training using only the seed samples.
Folder 'per_dataset' and 'per_dataset_stat'  contains a histogram of times when one of the compared methods worked better or worse/statistically better or worse divided by datasets.
Folder 'per_seeds' is the same as 'per_dataset', but divided by different no. seed samples per label used.