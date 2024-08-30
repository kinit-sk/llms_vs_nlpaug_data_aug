# LLMs vs Established Text Augmentation Techniques for Classification:
This repository contains both the data and code for this paper. This repository is structured as follows:

**Dataset folders (news_category, ag_news, atis, fb, yelp, sst5)**: each of the folders contains its own readme file with further instructions. In general, each folder contains collected data via LLM-based or established methods; scripts for collecting data and finetuning; and result for each finetuning done on the dataset per augmentation method, no. seed used, no. data collected and finetuned model used.

**aggregate_each_vs_each_{placeholder}.py**: these scripts are used for the aggregation of results from the finetunings of classifiers when comparing each type of classifier and finetuning method used.

**each_vs_each_{placeholder} folders**: these folders contain the aggregated results in form of csvs comparing each LLM-based method with each established augmentation method for downstream model accuracy and the scripts and results of visualization done on those results.   

**augnlp.ipynb**: jupyter notebook with examples of how we gather data from established augmentation methods

**requirements.txt**: python requirements for the scripts 


## Citing

