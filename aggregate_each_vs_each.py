import pandas as pd
import numpy as np
import scipy.stats as stats

def get_res(sample1, sample2):
#perform the Mann-Whitney U test
    value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided').pvalue

    if value > 0.05:
        return "Same dist.", False, value
    else:
        return "Different dist.", True, value
        

datasets = {'ag_news': 'news', 'news_category': 'news', 'atis': 'atis',  'fb': 'fb', 'sst': 'sst', 'yelp':'yelp'} 

llms_dct = {'llama': {0: 'roberta', 1: 'roberta_1', 2: 'roberta_2'}, 'gpt': {0: 'results', 1: 'results_1', 2: 'results_2'}}
llms_iter_dct = {'gpt': {0: '', 1: '', 2: ''}, 'llama': {0: '', 1: '_1', 2: '_2'}}

#datasets = {'sst_orig': 'sst'}
seeds = [5, 10, 20, 30, 40 ,50, 100]
atis_seeds = [5,10,20,25]
cols = [1,2,5,10,15]

class_methods  = ['cont_ins', 'back', 'cont', 'comb']
llm_methods = ['para', 'ins', 'swap']

def get_each_vs_each(class_methods, llm_methods):
    for llm in llms_dct.keys():
        dct_df_final = {'method':[], 'mean':[], 'std': [], 'is_stat': [], 'seed': [], 'rank':[], 'col':[], 'dataset': [], 'rand_seed': []}
        for rand_seed in range(0,3):
            for dataset_dir in datasets.keys():
                if dataset_dir == 'atis_orig':
                    seeds_to_iter = atis_seeds 
                else:
                    seeds_to_iter = seeds
                for seed in seeds_to_iter:
                    if dataset_dir == 'fb_orig' and seed == 100:
                        continue
                    dct_results = {}
                    
                    best_llm_method_val = np.array([0])
                    best_llm_method_name = None
                    best_llm_method_col = 0
                    
                    best_class_method_val = np.array([0])
                    best_class_method_name = None
                    best_class_method_col = 0
                
                    for col in cols:
                        for method in [class_methods]:
                            #print(llms_iter_dct[llm][rand_seed])
                            #print(method)
                            if method == 'comb':
                                df = pd.read_csv(dataset_dir+'/'+llms_dct['llama'][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_class.csv')
                            else:
                                if seed < 100:
                                    df = pd.read_csv(dataset_dir+'/'+llms_dct['llama'][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_class'+llms_iter_dct['llama'][rand_seed]+'.csv')
                                else:
                                    df = pd.read_csv(dataset_dir+'/'+llms_dct['llama'][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_class.csv')
                            
                            if best_class_method_val.mean() < df['res_orig'].values.mean():
                                best_class_method_val = df['res_orig'].values
                                best_class_method_name = method
                                best_class_method_col = col
                                    
                        for method in [llm_methods]:
                            if method == 'swap' and dataset_dir == 'news_category_orig':
                                df = pd.read_csv(dataset_dir+'/'+llms_dct[llm][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_llm.csv')
                            else:
                                if seed < 100:
                                    df = pd.read_csv(dataset_dir+'/'+llms_dct[llm][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_llm'+llms_iter_dct[llm][rand_seed]+'.csv')
                                else:
                                    df = pd.read_csv(dataset_dir+'/'+llms_dct[llm][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_llm.csv')

                            if best_llm_method_val.mean() < df['res_orig'].values.mean():
                                best_llm_method_val = df['res_orig'].values
                                best_llm_method_name = method
                                best_llm_method_col = col
                    
                    #print(col)
                    #print(dataset_dir)
                    #print(rand_seed)
                    #print(seed)
                    #print(llm)
                    is_stat = get_res(best_llm_method_val, best_class_method_val)
                #print(is_stat)
                #print(dataset_dir)
                #print(seed)
                
                    llm_rank =  1 if best_llm_method_val.mean() > best_class_method_val.mean() else 2
                    class_rank = 1 if best_class_method_val.mean() > best_llm_method_val.mean() else 2
       
                    dct_df_final['seed'].append(seed)
                    dct_df_final['method'].append(best_llm_method_name)
                    dct_df_final['rank'].append(llm_rank)
                    dct_df_final['mean'].append(best_llm_method_val.mean())
                    dct_df_final['std'].append(best_llm_method_val.std())
                    dct_df_final['is_stat'].append(is_stat[1])
                    dct_df_final['col'].append(best_llm_method_col)
                    dct_df_final['dataset'].append(dataset_dir)
                    dct_df_final['rand_seed'].append(rand_seed)
                    
                    dct_df_final['seed'].append(seed)
                    dct_df_final['method'].append(best_class_method_name)
                    dct_df_final['rank'].append(class_rank)
                    dct_df_final['mean'].append(best_class_method_val.mean())
                    dct_df_final['std'].append(best_class_method_val.std())
                    dct_df_final['is_stat'].append(is_stat[1])
                    dct_df_final['col'].append(best_class_method_col)
                    dct_df_final['dataset'].append(dataset_dir)
                    dct_df_final['rand_seed'].append(rand_seed)
            

        pd.DataFrame.from_dict(dct_df_final).to_csv('each_vs_each_roberta/'+llm_methods+'_vs_'+class_methods+'_'+llm+'.csv', index=False)

def get_each_vs_clean(method, is_llm):
    for llm in llms_dct.keys():
        dct_df_final = {'method':[], 'mean':[], 'std': [], 'is_stat': [], 'seed': [], 'rank':[], 'col':[], 'dataset': [], 'rand_seed': []}
        for rand_seed in range(0,3):
            for dataset_dir in datasets.keys():
                if dataset_dir == 'atis_orig':
                    seeds_to_iter = atis_seeds 
                else:
                    seeds_to_iter = seeds
                for seed in seeds_to_iter:
                    if dataset_dir == 'fb_orig' and seed == 100:
                        continue
                
                    dct_results = {}
                    
                    best_method_val = np.array([0])
                    best_method_name = None
                    best_method_col = 0
                
                    for col in cols:
                        if not is_llm:
                                if seed < 100:
                                    df = pd.read_csv(dataset_dir+'/'+llms_dct['llama'][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_class'+llms_iter_dct['llama'][rand_seed]+'.csv')
                                else:
                                    df = pd.read_csv(dataset_dir+'/'+llms_dct['llama'][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_class.csv')
                                if best_method_val.mean() < df['res_orig'].values.mean():
                                    best_method_val = df['res_orig'].values
                                    best_method_name = method
                                    best_method_col = col
                        else:    
                                if method == 'swap' and dataset_dir == 'news_category_orig':
                                    df = pd.read_csv(dataset_dir+'/'+llms_dct[llm][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_llm.csv')
                                else:
                                    if seed < 100:
                                        df = pd.read_csv(dataset_dir+'/'+llms_dct[llm][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_llm'+llms_iter_dct[llm][rand_seed]+'.csv')
                                    else:
                                        df = pd.read_csv(dataset_dir+'/'+llms_dct[llm][rand_seed]+'/'+datasets[dataset_dir]+'_'+method+'_seeds_'+str(seed)+'_coll_'+str(col)+'_llm.csv')
                                if best_method_val.mean() < df['res_orig'].values.mean():
                                    best_method_val = df['res_orig'].values
                                    best_method_name = method
                                    best_method_col = col
                    
                    clean_df = pd.read_csv(dataset_dir+'/'+llms_dct['llama'][rand_seed]+'/'+datasets[dataset_dir]+'_clean_seeds_'+str(seed)+'.csv')
                    clean_df_val = clean_df['res_orig'].values
                    is_stat = get_res(best_method_val, clean_df_val)       
                    
                    method_rank =  1 if best_method_val.mean() > clean_df_val.mean() else 2
                    clean_rank = 1 if clean_df_val.mean() > best_method_val.mean() else 2
       
                    dct_df_final['seed'].append(seed)
                    dct_df_final['method'].append(best_method_name)
                    dct_df_final['rank'].append(method_rank)
                    dct_df_final['mean'].append(best_method_val.mean())
                    dct_df_final['std'].append(best_method_val.std())
                    dct_df_final['is_stat'].append(is_stat[1])
                    dct_df_final['col'].append(best_method_col)
                    dct_df_final['dataset'].append(dataset_dir)
                    dct_df_final['rand_seed'].append(rand_seed)
                    
                    dct_df_final['seed'].append(seed)
                    dct_df_final['method'].append('clean')
                    dct_df_final['rank'].append(clean_rank)
                    dct_df_final['mean'].append(clean_df_val.mean())
                    dct_df_final['std'].append(clean_df_val.std())
                    dct_df_final['is_stat'].append(is_stat[1])
                    dct_df_final['col'].append(0)
                    dct_df_final['dataset'].append(dataset_dir)
                    dct_df_final['rand_seed'].append(rand_seed)
            

        pd.DataFrame.from_dict(dct_df_final).to_csv('each_vs_each_roberta/'+method+'_vs_clean_'+llm+'.csv', index=False)

for class_method in class_methods:
    for llm_method in llm_methods:
        get_each_vs_each(class_method, llm_method)

get_each_vs_clean('para', True)
get_each_vs_clean('swap', True)
get_each_vs_clean('ins', True)
get_each_vs_clean('cont', False)
get_each_vs_clean('cont_ins', False)
get_each_vs_clean('back', False)