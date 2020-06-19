# A textual analysis of Reddit mental health support Groups

## 1. Data
* 2018: Jan-April 2018
* 2019: Jan-April 2019
* pre: November 2018 to December 2019
* post: Jan-April 2020


## 2. Virtual Environment

All `.ipynb` can run on Google Colab
Install the `requirements.txt` to run the `.py`
* `conda create --name reddit --file requirements.txt`
* `conda activate reddit`


## Analyses
### Trend Analysis
* `reddit_descriptive.ipynb`

### Classification
* config.py #change paths
* `run_v8_<N>.sh` # runs `run.py` on cluster running each binary classifier on different nodes. N is the model (0=SGD L1, 1=SGD EN, 2=SVM, 3=ET, 4=XGB) 
* `run.py` # script to run binary classifiers
To summarize results, extract important features and test on COVID19_support (psychological profiler), run (change paths accordingly):
* classification_results.py


