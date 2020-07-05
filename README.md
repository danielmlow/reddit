# A textual analysis of Reddit mental health support groups

## 1. Data

Available at Open Science Framework (TODO): https://osf.io/7peyq/

Posts and text features for the following timeframes from 28 mental health and non-mental health subreddits:

 - 15 specific mental health complaints or communities (r/EDAnonymous, r/addiction,
   r/alcoholism, r/adhd, r/anxiety, r/autism, r/bipolarreddit, r/bpd,
   r/depression, r/healthanxiety, r/lonely, r/ptsd, r/schizophrenia,
   r/socialanxiety, and r/suicidewatch) 
- 2 broad mental health
   subreddits (r/mentalhealth, r/COVID19_support) 
- 11 non-mental health
   subreddits (r/conspiracy, r/divorce, r/fitness, r/guns, r/jokes,
   r/legaladvice, r/meditation, r/parenting, r/personalfinance,
   r/relationships, r/teaching). 

Downloaded using pushshift API. Re-use of this data is subject to Reddit API terms. Cite TODO if using this dataset.

Timeframes and correponding filenames:

* 2018 (Jan-April) `_2018_`   
* 2019 (Jan-April) `_2019_` (r/EDAnonymous appears)
* 2019 (December 2018 to December 2019) `_pre_` 
* 2020 (Jan-April) `_post_` r/COVID19_support appears 



## 2. Reproduce

All `.ipynb` can run on Google Colab for which data should be on Google Drive. 

To run the `.py`, create a virtual environment and install the `requirements.txt`:
* `conda create --name reddit --file requirements.txt`
* `conda activate reddit`

## Preprocessing
* `reddit_data_extraction.ipynb` download data
* `reddit_feature_extraction.ipynb` feature extraction for classification (TF-IDF was re-done separately on train set), trend analysis, and supervised dimensionality reduction.
* See below for preprocessing for topic modeling and unsupervised clustering 

## Analyses
##### Classification
* `config.py` set paths, subreddits to run, and sample size
* N is the model (0=SGD L1, 1=SGD EN, 2=SVM, 3=ET, 4=XGB)
* Run remotely: `run_v8_<N>.sh` runs `run.py` on cluster running each binary classifier on different nodes through `--job_array_task_id` set to one of range(0,15) 
* Run locally (set `--job_array_task_id` and `--run_modelN` accordingly): `python3 run.py --job_array_task_id=0 --run_version_number=8 --run_modelN=0`
* `classification_results.py`: figure 4-a, summarize results, extract important features, and visualize testing on COVID19_support (psychological profiler), run (change paths accordingly)

##### Trend Analysis
* `reddit_descriptive.ipynb`: figures 1 and 2

##### Topic Modeling
* `reddit_lda_pipeline.ipynb`: figure 3-a and 4-b

##### Unsupervised clustering
* `Unsupervised_Clustering_Pipeline.ipynb`: figures 3-b and 4-c

##### Supervised dimensionality reduction
* `reddit_cluster.ipynb`: figure 5
* `reddit_cluster.py`: UMAP on 50 random subsamples of 2019 (pre) data to determine sensor precision
    * run remotely: `run_umap.sh`
    * run locally (`--job_array_task_id` will run a single subsample): `python3 reddit_cluster.py --job_array_task_id=0 --plot=True --pre_or_post='pre'`
