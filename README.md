Data and code for "Natural language processing reveals vulnerable mental health groups and heightened health anxiety on Reddit during COVID-19" 

## 1. Data

Available at Open Science Framework (TODO): https://osf.io/7peyq/

**Citation:** 

(TODO) Natural language processing reveals vulnerable mental health groups and heightened health anxiety on Reddit during COVID-19. 

**License:** This dataset is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/
It was downloaded using pushshift API. Re-use of this data is subject to Reddit API terms. 

### 1.1. Reddit mental health dataset

find in `data/input/reddit_mental_health_dataset/`

Posts and text features for the following timeframes from 28 mental health and non-mental health subreddits:

 - **15 specific mental health support groups** (r/EDAnonymous, r/addiction,
   r/alcoholism, r/adhd, r/anxiety, r/autism, r/bipolarreddit, r/bpd,
   r/depression, r/healthanxiety, r/lonely, r/ptsd, r/schizophrenia,
   r/socialanxiety, and r/suicidewatch) 
- **2 broad mental health**
   subreddits (r/mentalhealth, r/COVID19_support) 
- **11 non-mental health
   subreddits** (r/conspiracy, r/divorce, r/fitness, r/guns, r/jokes,
   r/legaladvice, r/meditation, r/parenting, r/personalfinance,
   r/relationships, r/teaching). 

Downloaded using pushshift API. Re-use of this data is subject to Reddit API terms. Cite TODO if using this dataset.

`filenames` and corresponding timeframes:

- `post:` Jan 1 to April 20, 2020 (called "mid-pandemic" in manuscript; r/COVID19_support appears)
- `pre:` Dec 2018 to Dec 2019. A full year which provides more data for a baseline of Reddit posts
- `2019:` Jan 1 to April 20, 2019 (r/EDAnonymous appears). A control for seasonal fluctuations to match `post` data.
- `2018:` Jan 1 to April 20, 2018. A control for seasonal fluctuations to match `post` data.

See Supplementary Materials for more information.

Note: if subsampling (e.g., to balance subreddits), we recommend bootstrapping analyses for unbiased results. 


### 1.2. COVID-19 mention dataset (Figure 1)

find in `data/input/covid19_counts/`

Same posts as in `post` above for 15 mental health subreddits. 

Counting these tokens: `'corona','virus','viral','covid', 'sars','influenza','pandemic', 'epidemic', 
         'quarantine','lockdown', 'distancing', 'national emergency', 'flatten', 
             'infect','ventilator', 'mask','symptomatic',
            'epidemiolog', 'immun', 'incubation', 'transmission','vaccine'
`

* One column `covid19_boolean`: if one of these words appears at least once (Figure 1)
* One column `covid19_total`: total count of words
* One column `covid19_weighed_words`: total count of words normalized by the amount of words (n_words) in a post (Figure S3).  


### 2.3. COVID-19 cases
Confirmed COVID-19 cases obtained from ourworldindata.org/covid-cases (source: European CDC).


## 2. Reproduce

All `.ipynb` can run on Google Colab (for which data should be on Google Drive; code to load data from Google Drive is available in scripts) or on Jupter Notebook. 

To run the `.py` or `.ipynb` on Jupter Notebook, create a virtual environment and install the `requirements.txt`:
* `conda create --name reddit --file requirements.txt`
* `conda activate reddit`

### 2.1. Preprocessing
* `reddit_data_extraction.ipynb` download data
* `reddit_feature_extraction.ipynb` feature extraction for classification (TF-IDF was re-done separately on train set), trend analysis, and supervised dimensionality reduction.
* See below for preprocessing for topic modeling and unsupervised clustering 

### 2.2. Analyses
##### Classification
* `config.py` set paths, subreddits to run, and sample size
* N is the model (0=SGD L1, 1=SGD EN, 2=SVM, 3=ET, 4=XGB)
* Run remotely: `run_v8_<N>.sh` runs `run.py` on cluster running each binary classifier on different nodes through `--job_array_task_id` set to one of range(0,15) 
* Run locally (set `--job_array_task_id` and `--run_modelN` accordingly): 
```
python3 -i run.py --job_array_task_id=1 --run_modelN=0 --run_version_number=8 
```

* `classification_results.py`: figure 5-a, summarize results, extract important features, and visualize testing on COVID19_support (psychological profiler), run (change paths accordingly)

##### Trend Analysis
* `reddit_descriptive.ipynb`: figures 1 and 2

##### Topic Modeling
* `reddit_lda_pipeline.ipynb`: figure 4 and 5-b

##### Unsupervised clustering
* `Unsupervised_Clustering_Pipeline.ipynb`: figures 3 and 5-c

##### Supervised dimensionality reduction
* `reddit_cluster.ipynb`: figure 6
* `reddit_cluster.py`: UMAP on 50 random subsamples of 2019 (pre) data to determine sensor precision
    * run remotely: `run_umap.sh`
    * run locally (`--job_array_task_id` will run a single subsample): 
    ```
    python3 reddit_cluster.py --job_array_task_id=0 --plot=True --pre_or_post='pre'
    ```