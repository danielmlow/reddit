#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''


import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
import re

from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import switcher
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import FeatureUnion
# import config
import umap
sys.path.append('./../../catpro')
from catpro.preprocessing_text import extract_features
from catpro import data_helpers
# from catpro import evaluate_metrics


# from catpro.models import vector_models
# from catpro.models import lstm
from sklearn.model_selection import KFold
from hetero_feature_union import FeatureExtractor, ItemSelector
import parameters

seed_value= 1234

pd.options.display.width = 0

'''
from importlib import reload
reload(vector_models)
'''


def list_of_list_to_array(l):
	print(len(l))
	l1 = [n for i in l for n in i]
	l2 = np.array(l1)
	print(l2.shape)
	return l2


'''
# metrics
from sklearn.metrics import SCORERS
l = list(SCORERS.keys())
l.sort()
'''

# # Mount GDrive and attach it to the colab for data I/O
# from google.colab import drive
# drive.mount('/content/drive')
# data_folder = '/content/drive/My Drive/ML4HC_Final_Project/data/input/feature_extraction/'
# data_folder = './../../datum/reddit/input/feature_extraction/'

# subreddits = ['EDAnonymous', 'addiction', 'adhd', 'alcoholism', 'anxiety',
#  'bipolarreddit', 'bpd',  'depression',  'healthanxiety',
#        'jokes', 'legaladvice', 'meditation', 'mentalhealth',
#        'mentalillness', 'mindfulness', 'paranoia',
#        'personalfinance','ptsd', 'schizophrenia', 'socialanxiety',
#        'suicidewatch']





def subsample_df(df, subsample, overN):
	if df.shape[0] > overN:
		subsample_int = int(df.shape[0]*subsample)
	df = df.loc[np.random.choice(df.index,subsample_int, replace=False)]
	return df


def load_reddit(input_dir, subreddits, pre_or_post = 'pre', subsample=None, subsample_subreddits_overN=None, days = (0,-1)):

	# Careful: if you add COVID19_support and it does not exist in the first time step, then this will confuse metric learning
	reddit_data = pd.read_csv(input_dir +subreddits[0]+'_{}_features.csv'.format(pre_or_post), index_col=False)
	print('before:', subreddits[0], reddit_data.shape)
	if subsample:
		reddit_data = subsample_df(reddit_data, subsample, subsample_subreddits_overN)
		print('after:', subreddits[0], reddit_data.shape)

	for i in np.arange(1, len(subreddits)):
		new_data = pd.read_csv(input_dir+subreddits[i]+'_{}_features.csv'.format(pre_or_post))
		print('before:',subreddits[i], new_data.shape)
		if subsample:
			new_data = subsample_df(new_data, subsample, subsample_subreddits_overN)
		print('after:',subreddits[i], new_data.shape)
		reddit_data = pd.concat([reddit_data, new_data], axis=0)

	return reddit_data




def heatmap_probs(matrix, subreddits, days):
	'''
	:param df:
	:param subreddits:
	:param days:
	:return:

	# Simulation:

	import random
	df = []
	for i in np.arange(0.005,0.5,0.05)[:len(rows)]:
		print(i)
		df.append(list(np.random.normal(i, 0.1, size=len(cols))))

	'''
	import seaborn as sns
	import matplotlib.pyplot as plt

	df = pd.DataFrame(matrix, index=subreddits, columns=days)
	# # simulation
	# df.iloc[4,11:] = np.random.normal(0.6, 0.05, size=8)
	# df.iloc[-1,11:] = np.random.normal(0.2, 0.05, size=8)

	# clean
	cols = list(df.columns)
	cols = [n.replace('2020/', '') for n in cols]
	df.columns = cols

	sns.heatmap(df)
	plt.tight_layout()
	plt.savefig('./data/toy_prediction.png', epi=200)
	return df

'''
from importlib import reload
reload(config)

'''

#
# def stemming_tokenizer(str_input):
# 	'''
# 	http://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/counting-and-stemming/
#
# 	:param str_input:
# 	:return:
# 	'''
# 	words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
# 	words = [porter_stemmer.stem(word) for word in words]
# 	return words


if __name__ == "__main__":
	# Config
	import config
	input_dir = config.input_dir
	output_dir = config.output_dir
	# hyperparams = config.hyperparams
	model = config.model

	run_version_number = config.run_version_number
	subreddits = config.subreddits
	cv = int(config.cv)

	subsample = int(config.subsample)
	include_subreddits_overN = int(config.include_subreddits_overN)

	run_modelN = int(config.run_modelN)
	# mkdir output dir and logger
	run_final_model = config.run_final_model

	dim_reduction = config.dim_reduction

	if run_final_model:
		output_dir = data_helpers.make_output_dir(output_dir, name='run_final_model_v{}_model{}'.format(run_version_number, run_modelN))
	else:
		output_dir = data_helpers.make_output_dir(output_dir, name='run_gridsearch_v{}_model{}'.format(run_version_number, run_modelN))


	# Load data
	if model in ['lstm', 'gru', 'bi-lstm', 'bi-gru']:
		# todo
		pass
	else:
	# 	vector models
		reddit_data = load_reddit(input_dir+'feature_extraction/', subreddits, pre_or_post = 'pre')

	days = np.unique(reddit_data.date)
	days.sort()
	days_train = days[:]
	reddit_data = reddit_data [reddit_data .date.isin(days_train)]
	counts = reddit_data.groupby(["subreddit", "date"]).size().reset_index(name='count')
	sr_all = []
	counts_all = []
	for sr in subreddits:
		counts_d = counts[counts.subreddit == sr].sum()
		print(sr, ': ', np.round(float(list(counts_d)[-1]), 2))
		sr_all.append(sr)
		counts_all.append(np.round(float(list(counts_d)[-1]), 2))


	# Exclude under
	if include_subreddits_overN:
		subreddits = [n for n,i in zip(sr_all, counts_all) if i>include_subreddits_overN ]


	for sr in subreddits:
		reddit_data = reddit_data[reddit_data.subreddit.isin(subreddits)]


	# Features

	features = list(reddit_data.columns)
	features = [n for n in features if n not in ['subreddit', 'author', 'date', 'post']]
	print('double check features: ', features)
	posts = np.array(reddit_data.subreddit.value_counts()).astype(int)
	days = np.unique(reddit_data.date)



	# Build X
	# docs =· todo for tfidf
	docs_all = []
	X = []
	y = []
	for sr in subreddits:
		df_subreddit = reddit_data[reddit_data.subreddit==sr]
		if subsample:
			df_subreddit = df_subreddit.sample(n=subsample, random_state=seed_value)
		df_subreddit_X = df_subreddit[features].values
		df_subreddit_y = list(df_subreddit .subreddit)

		docs = list(df_subreddit['post'])
		docs = [post.replace('\n\n', ' ').replace('  ', ' ').replace('“', '').replace('”', '') for post in
		         docs]  # here I remove paragraph split, double spaces and some other weird stuff, this should be done once for all posts\n",



		X.append(df_subreddit_X)
		y.append(df_subreddit_y)
		docs_all.append(docs)




	X, y, docs_all = list_of_list_to_array(X),list_of_list_to_array(y),list_of_list_to_array(docs_all)
	le = preprocessing.LabelEncoder()
	y_encoded = le.fit_transform(y)

	# Split
	X_train, X_test, y_train, y_test, docs_train, docs_test  = train_test_split(X, y_encoded, docs_all,test_size=0.20, random_state=seed_value)


	# from importlib import reload
	# reload(extract_features)
	train_tfidf, test_tfidf, feature_names_tfidf = extract_features.tfidf(X_train_sentences=docs_train, X_test_sentences=docs_test,
	                                                                      ngram_range=(1, 2),
	                                                                      max_features=256, min_df=2, max_df=0.8,
	                                                                      model=model)

	X_train = np.concatenate([X_train, train_tfidf], axis=1)
	X_test = np.concatenate([X_test, test_tfidf], axis=1)

	# d = {'X': X_train, 'docs': docs_train}
	# # ItemSelector(key='docs').fit_transform(d)


	# Run models
	if run_final_model:
		parameters = parameters.parameters_all_models_final(y,dim_reduction)
	else:
		parameters = parameters.parameters_all_models(y, dim_reduction=dim_reduction)

	# write all variables in config)
	with open(output_dir + 'config.txt', 'a+') as f:
		f.write(str(subreddits))
		f.write('\n')
		f.write(str(parameters))
		f.write('\n')

	if dim_reduction:
		pipeline = Pipeline([
			('normalization', None),
			('umap', umap.UMAP(n_components=2,min_dist=0.1,  metric='correlation', random_state=seed_value)),
			('clf', switcher.ClfSwitcher()),
		])


	else:
		pipeline = Pipeline([
			('normalization', None),
			('feature_selection', SelectKBest()),
			('clf', switcher.ClfSwitcher()),
		])

	if run_final_model:
		# TODO would this work model_and_params = parameters[run_modelN]
		for i, model_and_params in enumerate(parameters):
			if i!= run_modelN:
				continue

			pipeline.set_params(**model_and_params)
			pipeline.fit(X_train, y_train)
			y_pred = pipeline.predict(X_test)
			# Evaluate
			report = classification_report(y_test, y_pred, output_dict=True)
			df = pd.DataFrame(report).transpose()


			model_name = str(model_and_params.get('clf__estimator')).split('(')[0]
			df.to_csv(output_dir+'report_{}.csv'.format(model_name),index_label=0)
			df.to_latex(output_dir+'report_latex_{}'.format(model_name))
			with open(model_name+'_params.txt', 'a+') as f:
				f.write(str(model_and_params))



	else:
		# Hyperparameter tuning
		# models_all = []
		# results_all = []
		# best_params_all = []
		# best_score_all = []

		for i, model_and_params in enumerate(parameters):
			if i!= run_modelN:
				continue

			gscv = GridSearchCV(pipeline, model_and_params, cv=cv, n_jobs=-1, return_train_score=False, verbose=0,
		                    scoring='f1_weighted')
			gscv.fit(X_train, y_train)


			results = pd.DataFrame(gscv.cv_results_)
			print('=======================================================\n')

			print(gscv.best_params_)
			print(gscv.best_score_)
			print('=======================================================\n')


			# models_all.append(gscv)
			# results_all.append(results)
			# best_params_all.append(gscv.best_params_)
			# best_score_all.append(gscv.best_score_)


			model_name= str(results.param_clf__estimator[0]).split('(')[0]

			# joblib.dump(gscv.best_estimator_, output_dir + '{}.pkl'.format(model_name))

			with open(output_dir+model_name+'.txt', 'a+') as f:
				f.write('\n=======================================================\n')
				f.write(str(gscv.best_estimator_))
				f.write('\n')
				f.write(str(np.round(gscv.best_score_,4)))
				f.write('\n=======================================================\n')


			# joblib.dump(gscv.best_estimator_, output_dir+'{}.pkl'.format(model_name))
			results.to_csv(output_dir+model_name+'.csv',index_label=0)




# 		#todo:



	# Todo: fix so tfidf wont overfit on within training data
	# ================================================================================
	# X_docs_train = np.append(X_train, np.reshape(docs_train, (docs_train.shape[0], 1)), axis=1)
	# combined_features = ('union', FeatureUnion(
	# 	transformer_list=[
	# 		# Pipeline for pulling pre-extracted features "X"
	# 		('X_features', ItemSelector('X')),
	#
	# 		# Pipeline for extracting tfidf from clean "docs" (strings)
	# 		('tfidf', Pipeline([
	# 			('selector', ItemSelector('docs')),
	# 			('tfidf_vec', TfidfVectorizer(ngram_range = (1, 2), max_features = 256, min_df = 2, max_df = 0.8, stop_words='english', tokenizer = stemming_tokenizer)),
	# 							])
	# 		 ),
	# 	],
	# ))
	#
	#
	# pipeline = Pipeline([
	# 	('subjectbody', FeatureExtractor()),
	# 	(combined_features),
	# 	('normalization', None),
	# 	('feature_selection', SelectKBest()),
	# 	('clf', switcher.ClfSwitcher()),
	# ])

	# ================================================================================

	#
	#
	# 	# weight components in FeatureUnion
	# 	transformer_weights={
	# 		'subject': 0.8,
	# 		'body_bow': 0.5,
	# 		'body_stats': 1.0,
	# 	},
	# )),







#
#
# pipeline = Pipeline([
#   ('extract_essays', EssayExractor()),
#   ('features', FeatureUnion([
#     ('ngram_tf_idf', Pipeline([
#       ('counts', CountVectorizer()),
#       ('tf_idf', TfidfTransformer())
#     ])),
#     ('essay_length', LengthTransformer()),
#     ('misspellings', MispellingCountTransformer())
#   ])),
#   ('classifier', MultinomialNB())
# ])


# https://scikit-learn.org/0.19/auto_examples/hetero_feature_union.html
#
# combined_features = ([
#     ('tfidf', Pipeline([
#       ('counts', CountVectorizer()),
#       ('tf_idf', TfidfTransformer()),
# 	('pre-extractednormalization', None)
#     ])),
#
# pipeline = Pipeline([
#
# 	('features', combined_features),
# 	('feature_selection', SelectKBest()),
#     ('clf', switcher.ClfSwitcher()),
# ])


