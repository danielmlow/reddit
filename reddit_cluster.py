#!/usr/bin/env python
# coding: utf-8
'''
Author: Daniel M. Low
License: Apache 2.0


'''

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from datetime import datetime
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore
from scipy import stats
import itertools


# In[2]:



def str2boolean(s):
	if s == 'False':
		s_new = False
	else:
		s_new = True
	return s_new


# In[47]:


# Config

toy=False
sample = 0
pre_or_post = 'pre'
plot=False

timestamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
# data_folder = '/content/drive/My Drive/ML4HC_Final_Project/data/input/'
data_folder = './../../datum/reddit/input/reddit_mental_health_dataset/'
output_dir = f'./../../datum/reddit/output/supervised_umap/umap_cluster_{pre_or_post}_{timestamp}/'
print(output_dir)


# In[48]:


# # # Mount GDrive and attach it to the colab for data I/O
# from google.colab import drive
# drive.mount('/content/drive')


# In[49]:


try: os.mkdir(output_dir)
except: pass


# In[50]:


# Or load from arguments
parser = argparse.ArgumentParser()
parser.add_argument('--job_array_task_id',
                    help='default: ${SLURM_ARRAY_TASK_ID} or 1. When using job arrays, this will be set by the bash script by ${SLURM_ARRAY_TASK_ID} or set to 1, which will be substracted below by 1 for zero indexing')
parser.add_argument('--plot', help='plot of each run')
parser.add_argument('--toy', help='run quickly with less labels, parameters and splits')
parser.add_argument('--pre_or_post', help='post, pre, 2019, 2018')

args = parser.parse_args()
if args.job_array_task_id != None:
	sample = int(args.job_array_task_id) - 1
    
if args.plot!=None:
	plot = str2boolean(args.plot)
    
if args.toy!=None:
	toy = str2boolean(args.toy)

if args.toy!=None:
	pre_or_post = str(args.pre_or_post)

    


# In[4]:



seed_value = None
# silence NumbaPerformanceWarning
import warnings
import numba
from numba.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


# In[12]:


import pickle
def save_obj(obj, path ):
    with open(path , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    return a[p], b[p], c[p]


# In[7]:


subreddits = ['EDAnonymous',
 'addiction',
 'adhd',
 'alcoholism',
 'anxiety',
 'autism',
 'bipolarreddit',
 'bpd',
 'depression',
 'healthanxiety',
 'lonely',
 'ptsd',
 'schizophrenia',
 'socialanxiety',
 'suicidewatch']
# In[13]:


def scatter_plot(X_reduced, y, color_code, method, annotate = False, title = 'title', savefig_path = False,
                 centers= None, dists=None):

    plt.clf()
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.figure(figsize=(12,12), dpi= 80, facecolor='w', edgecolor='k')
    palette =   sns.color_palette("husl", len(np.unique(y)))
    # Shuffle
    X_reduced, y, color_code = unison_shuffled_copies(X_reduced, y, color_code)
    
    data = pd.DataFrame(X_reduced, columns = ['x', 'y'])
    data['label'] = y
    color_order = np.unique(y)
    print(color_order)
#     color_order.sort()

    g = sns.lmplot(data=data, x='x', y='y', hue='label', hue_order=color_order,
                   fit_reg=False, legend=True,palette=palette, legend_out=True, scatter_kws={"s": 10})
    
    if annotate:
        for i, txt in enumerate(annotate_names):
            plt.annotate(txt, (X_reduced[:, 0][i], X_reduced[:, 1][i]))

    # Centroids
    plt.scatter(
        centers[:, 0], centers[:, 1],
        s=10, marker='x',
        c='magenta', edgecolor='black',
        label='centroids')
    if title:
        plt.title(title)
    
    if savefig_path:
        print('plotting...')
        plt.savefig(savefig_path+title+'.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


# In[14]:


def run_umap(X=None, y=None, method = 'unsupervised', scaler=None, neighbor = 10, dist=0.1, metric='correlation', 
             color_code = None, annotate_names = None, annotate = False, test_set = True, title=None, 
             savefig_path = False, X_test=None, y_test=None, color_code_test = None, plot=True):
    
    reducer = umap.UMAP(n_components=dimension, n_neighbors = neighbor, min_dist=dist,metric=metric,random_state=seed_value) #, TSNE(n_components=k, random_state=seed_value), PCA(n_components=k, random_state=seed_value)]
    reducer_name = 'umap' #, 'tsne', 'pca']

    pipeline = Pipeline([
        ('normalization', scaler),
        ('reducer', reducer), ])

    y_encoded = LabelEncoder().fit_transform(y)
    if method == 'supervised':
        X_reduced = pipeline.fit_transform(X, y_encoded)
    elif method == 'metric_learning':
        X_reduced = pipeline.fit_transform(X, y_encoded)
        X_reduced_test = pipeline.transform(X_test)
        
    elif method == 'unsupervised':
        X_reduced = pipeline.fit_transform(X)
    
    print('running kmeans...')
    # Set k to amount of subreddits
    k = len(np.unique(y))
    # Fit kmeans
    km = KMeans(n_clusters = k, random_state = seed_value).fit(X_reduced)
    # Obtain euclidean distance between centroids
    centers = km.cluster_centers_
    # find centroid labels      
    closest, _ = pairwise_distances_argmin_min(centers, X_reduced)
    data = pd.DataFrame(X_reduced, columns = ['x1', 'x2'])
    data['label'] = y
    centers_labels = list(data.loc[closest].label)
    
    # Plot in 2D
    if plot:
        assert dimension == 2 
        if method == 'metric_learning':
            # train: first time point
            scatter_plot(X_reduced, y, color_code, method, annotate = annotate, title = 'First time step (train set)', savefig_path = savefig_path )
            # test: next time points            
            scatter_plot(X_reduced_test, y_test, color_code_test, method, annotate = annotate, title = title, savefig_path = savefig_path )
            
        else:
            scatter_plot(X_reduced, y, color_code, method, annotate = annotate, title = title, savefig_path = savefig_path, centers=centers)
    if method == 'metric_learning':
        return X_reduced, X_reduced_test
    else:
        return X_reduced, centers, centers_labels

def scatter_X_reduced(X_reduced, color_code, annotate_names):
    plt.clf()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color_code, alpha=0.8, s=6)
    if annotate:
        for i, txt in enumerate(annotate_names):
            plt.annotate(txt, (X_reduced[:, 0][i], X_reduced[:, 1][i]))
    plt.show()
    
    
def gridsearch_sets(metrics, n_neighbors, min_dist, n_dimensions, scalers, repeat):
    gridsearch  = []
    for metric in metrics:
        for neighbor in n_neighbors:
            for dist in min_dist:
                for dimension in n_dimensions:
                    for scaler in scalers:
                        for r in repeat:
                            gridsearch.append([metric,neighbor,dist,dimension,scaler, r])
    return gridsearch


# ## Load data with our without subsampling

# In[55]:


def subsample_df(df, subsample):
	if type(subsample) == float:
		subsample = int(df.shape[0]*subsample)
	df = df.reset_index(drop=True)
	df2 = df.loc[np.random.choice(df.index,subsample, replace=False)]
	return df2


# In[56]:


def clean(df):
    # remove author duplicates and shuffle so we dont keep only first posts in time
    reddit_data = df.sample(frac=1) #shuffle
    reddit_data = reddit_data.drop_duplicates(subset='author', keep='first')
    reddit_data  = reddit_data [~reddit_data.author.str.contains('|'.join(['bot', 'BOT', 'Bot']))] # There is at least one bot per subreddit
    reddit_data = reddit_data[~reddit_data.post.str.contains('|'.join(['quote', 'QUOTE', 'Quote']))] # Remove posts in case quotes are long
    reddit_data = reddit_data.reset_index(drop=True)
    return reddit_data


def load_reddit(subreddits, data_folder='./', subsample = 5600,pre_or_post = 'pre'):
    # subsample = 5600 #False, int for balanced, or 0.1 for unbalanced proportion, 5600
    # Careful: if you add COVID19_support and it does not exist in the first time step, then this will confuse metric learning
    subreddits.sort()
    # Main features
    # Load first subreddit to build DF
    reddit_data = pd.read_csv(data_folder+subreddits[0]+'_{}_features_tfidf_256.csv'.format(pre_or_post), index_col=False)        
    #     Clean
    reddit_data = clean(reddit_data)
    # remove jan and feb data from covid19_support because there's not enough data and if not kmeans will assign two centroids to another larger subreddit
    days = np.unique(reddit_data.date)
    days_jan_feb = [n for n in days if '2020/01' in n or '2020/02' in n]
    days_jan_feb
    if subreddits[0]=='COVID19_support' and pre_or_post == 'post':
      reddit_data = reddit_data[~reddit_data.date.isin(days_jan_feb)]

    # Subsample to int or subsample float
    print(reddit_data.shape)
    if subsample and subreddits[0] !='COVID19_support':
        reddit_data = subsample_df(reddit_data, subsample)
        print(reddit_data.shape)

    # Add next subreddits
    for i in np.arange(1, len(subreddits)):
        print('===')
        print(subreddits[i])
        new_data = pd.read_csv(data_folder +subreddits[i]+'_{}_features_tfidf_256.csv'.format(pre_or_post), index_col=False)
        # Clean
        new_data  = clean(new_data )
        if subreddits[i]=='COVID19_support' and pre_or_post == 'post':
              reddit_data = reddit_data[~reddit_data.date.isin(days_jan_feb)]
        print(new_data.shape)
        if subsample and subreddits[i] !='COVID19_support':
            new_data = subsample_df(new_data, subsample)
            print(new_data.shape)
        reddit_data = pd.concat([reddit_data, new_data], axis=0)

    return reddit_data


# In[27]:


# reddit_data = load_reddit(subreddits, data_folder=data_folder, subsample = 5600,pre_or_post = 'pre')

# counts = reddit_data.groupby(["subreddit", "date"]).size().reset_index(name='count')
# print('counts per day:')
# for sr in subreddits:
#   counts_d = counts[counts.subreddit == sr].mean()
#   print(sr, ': ', np.round(float(counts_d),2))


# ## Find optimal params (optimize on silhouette score)

# In[28]:


from scipy.spatial.distance import directed_hausdorff, euclidean


def hausdorff_distances(X,y, labels):
    len_labels = len(labels)
    # build empty df
    pairwise_distances_hausdorff = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)
    # pairwise_distances_euclidean = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)                          

    # Build df out of X
    df = pd.DataFrame(X)
    df.columns = ['x1', 'x2']
    df['label'] = y

    # Compute pairwise distance between labelled arrays 
    for row in range(len_labels):
        for col in range(len_labels):
            label_a = labels[row]
            label_b = labels[col]
            label_a_values = df[df.label==label_a][['x1','x2']].values
            label_b_values = df[df.label==label_b][['x1','x2']].values
            dist_hausdorff = directed_hausdorff(label_a_values,label_b_values)
            pairwise_distances_hausdorff.iloc[row,col]= dist_hausdorff[0]
    #         dist_euclidean = euclidean(label_a_values.mean(axis=0),label_b_values.mean(axis=0))
    #         pairwise_distances_euclidean.iloc[row,col]= dist_euclidean


    return pairwise_distances_hausdorff


# ## Run umap hyperparameter tuning OR 50 sample on 2019 data 

# In[ ]:


# I downloaded this script as .py and ran on cluster.  


# In[ ]:


# Load data
if pre_or_post == 'post':
    # Jan-April 2020
    subsample = 1350 #can set to False
    subreddits_edited = subreddits.copy()
    # todo: this will distort error from 2019:     
    #     subreddits_edited = ['COVID19_support']+subreddits


elif pre_or_post == 'pre':
    # all of 2019
    subsample = 2700
    subreddits_edited = subreddits.copy()

elif pre_or_post == '2019':
    # jan-april 2019     
    subsample = 900
    subreddits_edited = subreddits.copy()

elif pre_or_post == '2018':
    # jan-april 2018
    subsample = 450
    subreddits_edited = subreddits.copy()
    subreddits_edited.remove('EDAnonymous')

# Load data: define sample size
repeat_n = 50
sample_names = [f'{subsample}_{n}' for n in range(repeat_n)]

if toy:
    subsample = 20
    repeat_n = 2
    sample_sizes = [subsample] * repeat_n
    sample_names = [f'{subsample}_{n}' for n in range(repeat_n)]
    
#  sample is defined through jobid from argparse 
name = sample_names[sample]
savefig_path = output_dir+f'run_{name}_'
reddit_data = load_reddit(subreddits_edited, data_folder=data_folder, subsample = subsample,pre_or_post = pre_or_post)


# In[5]:


# Run umap hyperparameter tuning OR 50 sample on 2019 data
# for i, name in zip(sample_sizes, sample_names):
print('\n\n{} =========================================='.format(subsample))
results = {}
results_i = []
results_i_dists = [] #euclidean
results_i_dists_hausdorff = []
results_i_centers = []
# reddit_data = load_reddit(subreddits, data_folder=data_folder, subsample = i,pre_or_post = 'pre')
features = list(reddit_data.columns)
features = [n for n in features if n not in ['subreddit','author','date','post']]
#   print('double check features: ',features)
X = reddit_data[features].values
y = reddit_data.subreddit.values

method = 'supervised'# 'metric_learning', 'supervised', 'unsupervised'
annotate = False
annotate_names = False
run_Ntimesteps = 1# len(X)#len(X) #1,2 ... len(X) 
color_code = y.copy()


# Set up gridsearch
n_dimensions =  [2,] #[4,8,16,32,64,128] # https://umap-learn.readthedocs.io/en/latest/parameters.html
n_neighbors = [200] #[50,100,200] # CANNOT USE MORE THAN N participants, default=10 Lower values seem to work better in 2D. This means that low values of n_neighbors will force UMAP to concentrate on very local structure
min_dist = [0]  # [0,0.1,0.5,0.99] # default=0.1. Lower values seem to work better in 2D. controls how tightly UMAP is allowed to pack points together. Larger values of min_dist will prevent UMAP from packing point together and will focus instead on the preservation of the broad topological structure instead.
metrics = ['cosine'] # ['euclidean','cosine'] 
repeat_n = 1
repeat = [1]*repeat_n # to check how much randomness is affecting results, one can repeat
scalers = [StandardScaler()]

gridsearch= gridsearch_sets(metrics, n_neighbors, min_dist, n_dimensions, scalers, repeat)
for j, (metric,neighbor,dist,dimension,scaler, r) in enumerate(gridsearch):
    print('\n{} out of {}'.format(j, len(gridsearch)))
    print('====metric: {},  {} neighbor (low=maintain local structure),  {} dist (low=tightly packed),  {}D======'.format(metric,neighbor,dist,dimension))
    title = '{}_metric_{}_neighbor_{}_dist'.format(metric,neighbor,dist)
    print('running umap...')         
    X_reduced, centers, centers_labels  = run_umap(X = X, y = y, method = method, scaler=  scaler, neighbor = neighbor , dist=dist, metric=metric, 
        color_code = color_code, annotate_names = annotate_names, annotate = annotate, title=title,
        savefig_path = savefig_path, plot = plot)

    print('runnning metrics...')
    # compute euclidean distances
    dists = euclidean_distances(centers)
    dists_df = pd.DataFrame(dists, columns = centers_labels, index=centers_labels)
    # hausdorff distance
    
    dists_df_hausdorff = hausdorff_distances(X_reduced,y, subreddits)
    # Compute silhouette score
    sil_score = silhouette_score(X_reduced, y)
    
    # centers
    df_centers = pd.DataFrame(centers)
    df_centers['labels'] = centers_labels     

    # Compute convex hull
    hull = ConvexHull(centers)
    hull_area = hull.volume #volume is area in 2D
    
    #     Append
    results_i.append([metric,neighbor,dist,sil_score, hull_area])
    results_i_dists.append(dists_df)
    results_i_dists_hausdorff.append(dists_df_hausdorff)
    results_i_centers.append(df_centers)


results_gs = pd.DataFrame(results_i)
results_gs.columns = ['metric', 'neighbor','dist', 'sil_score','convexhull']
results_gs = results_gs.sort_values('sil_score')
# results[name] = results_gs
timestamp = datetime.now().isoformat()
results_gs.to_csv(output_dir+f'run_{name}_umap_gs_params_scores_{timestamp}.csv')
# euclidean
results_gs_dists = pd.concat(results_i_dists)
results_gs_dists.to_csv(output_dir+f'run_{name}_umap_gs_dists_{timestamp}.csv', )
# hausdorff
results_gs_dists_hausdorff = pd.concat(results_i_dists_hausdorff)
results_gs_dists_hausdorff.to_csv(output_dir+f'run_{name}_umap_gs_dists_hausdorff_{timestamp}.csv', )
# centers
results_gs_centers = pd.concat(results_i_centers)
results_gs_centers.to_csv(output_dir+f'run_{name}_umap_gs_centers_{timestamp}.csv', )
print('done.')
