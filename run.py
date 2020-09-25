#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''


import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import switcher
import umap
# Local scripts
# Clone catpro from https://github.com/danielmlow/catpro/
sys.path.append('./../../catpro') #Change dir to cloned dir
from catpro.preprocessing_text import extract_features
from catpro import data_helpers
import config_parameters
import load_reddit

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


def final_model(X_train, y_train, X_test, y_test,run_modelN, subreddit, subreddits,features,output_dir, parameters=None,append_to_name=None):
	pipeline = config_parameters.final_pipeline(run_modelN)
	# TODO would this work model_and_params = parameters[run_modelN]

	# todo for gridsearch
	# for i, model_and_params in enumerate(parameters):
	# 	if i != run_modelN:
	# 		continue
		# model_name = str(model_and_params.get('clf__estimator')).split('(')[0]

		# pipeline.set_params(**model_and_params)
	pipeline.fit(X_train, y_train)
	y_pred = pipeline.predict(X_test)
	# Evaluate
	report = classification_report(y_test, y_pred, target_names=subreddits, output_dict=True)
	df = pd.DataFrame(report).transpose()

	model_name = str(pipeline['clf']).split('(')[0]
	model_name = model_name.replace('[','')

	if append_to_name:
		model_name += '_'+append_to_name

	df.to_csv(output_dir + 'report_{}.csv'.format(model_name), index_label=0)
	df.to_latex(output_dir + 'report_latex_{}'.format(model_name))

	if 'SVC' in model_name or 'SVM' in model_name or 'SGD' in model_name:
		coefs = pipeline['clf'].coef_
		coefs_df = pd.DataFrame(coefs).T
		# coefs_df = pd.concat([pd.DataFrame(features), pd.DataFrame(np.transpose(coefs))], axis=1)
		if len(subreddits)<3:
			coefs_df.columns = [subreddit]
		else:
			coefs_df.columns = subreddits
		coefs_df.index = features
		coefs_df.to_csv(output_dir + 'coefs_df_{}.csv'.format(model_name))


		if len(subreddits)>2:
			for sr in subreddits:
				coef_sr = coefs_df.sort_values([sr])
				coef_sr.to_csv(output_dir + 'coefs_df_{}_{}.csv'.format(model_name, sr))
				with open(output_dir + 'coefs_df_{}_summary.txt'.format(model_name), 'a+') as f:
					f.write('\n\n==========================================\n')
					f.write('\n{} top: \n'.format(sr))
					f.write(str(coef_sr[sr].iloc[-20:]))
					f.write('=========\n')
					f.write('\n{} bottom: \n'.format(sr))
					f.write(str(coef_sr[sr].iloc[:10]))


		y_pred_probs = pipeline.predict_proba(X_test)
		y_pred_probs = pd.DataFrame(y_pred_probs)
		y_pred_probs.columns = subreddits
		y_pred_probs.to_csv(output_dir + 'y_pred_probs_{}.csv'.format(model_name), index=None)


	cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test), sample_weight=None)

	pd.DataFrame(cm).to_csv(output_dir + 'confusion_matrix_{}.csv'.format(model_name))
	# plot_outputs.plot_confusion_matrix(cm, subreddits, normalize=True, save_to=output_dir + 'confusion_matrix.png')



	# Features
def df_to_X(reddit_data, task='binary'):
	features = list(reddit_data.columns)
	features = [n for n in features if n not in ['subreddit', 'author', 'date', 'post']]
	print('double check features: ', features)

	# Build X
	docs_all = [] #for tfidf
	X = []
	y = []
	subreddits = np.unique(list(reddit_data.subreddit))
	print(subreddits)
	for sr in subreddits:
		df_subreddit = reddit_data[reddit_data.subreddit==sr]
		# if subsample:
		# 	df_subreddit = df_subreddit.sample(n=subsample, random_state=seed_value)
		df_subreddit_X = df_subreddit[features].values
		df_subreddit_y = list(df_subreddit .subreddit)

		docs = list(df_subreddit['post'])
		docs = [post.replace('\n\n', ' ').replace('  ', ' ').replace('“', '').replace('”', '') for post in
		         docs]  # here I remove paragraph split, double spaces and some other weird stuff, this should be done once for all posts\n",

		X.append(df_subreddit_X)
		y.append(df_subreddit_y)
		docs_all.append(docs)


	X, y, docs_all = list_of_list_to_array(X),list_of_list_to_array(y),list_of_list_to_array(docs_all)

	# Make sure 'control' is always 0
	try:
		y = np.array([n.replace('control', '0') for n in y])
	except: pass
	le = preprocessing.LabelEncoder()
	y_encoded = le.fit_transform(y)

	# Split
	X_train, X_test, y_train, y_test, docs_train, docs_test  = train_test_split(X, y_encoded, docs_all,test_size=0.20, random_state=seed_value)
	return X_train, X_test, y_train, y_test, docs_train, docs_test, features


def df_to_X_midpandemic(df, timestep = None,filter_days = ['2020/03/11', '2020/04/20'], subreddit = 'COVID19_support'):


	features = list(df.columns)
	features = [n for n in features if n not in ['subreddit', 'author', 'date', 'post']]

	if filter_days:
		df.date = df.date.replace({'/': '-'}, regex=True)

		start_date = filter_days[0].replace('/','-')
		end_date = filter_days[1].replace('/', '-')
		df = df[(df['date'] > start_date ) & (df['date'] < end_date )]

	df_sr = df[df.subreddit == subreddit]
	df_control = df[df.subreddit != subreddit]
	df_control = load_reddit.subsample_df(df_control,df_sr.shape[0])
	df_control.subreddit = 'control'

	df_sr = pd.concat([df_sr,df_control])

	X_test_sr = df_sr[features].values
	y_test_sr = list(df_sr .subreddit)
	docs_test_sr = list(df_sr ['post'])
	docs_test_sr = [post.replace('\n\n', ' ').replace('  ', ' ').replace('“', '').replace('”', '') for post in
	                docs_test_sr]  # here I remove paragraph split, double spaces and some other weird stuff, this should be done once for all posts\n",

	le = preprocessing.LabelEncoder()

	# Make sure 'control' is always 0
	try:
		y_test_sr  = np.array([n.replace('control', '0') for n in y_test_sr ])
	except: pass

	y_test_sr = le.fit_transform(y_test_sr )


	if timestep:
		# todo
		days = np.unique(df.date)
		days_timestep = days[::timestep]
		X = []
		y = []
		for i in range(0, len(days), timestep):
			days_week = days[i:i + timestep]
			df_week = df[df.date.isin(days_week)]
			df_week_feature_cols = df_week[features].values
			df_week_y = list(df_week.subreddit)
			X.append(df_week_feature_cols)
			y.append(df_week_y)


	if [0] == list(np.unique(y_test_sr)):
		y_test_sr = [1]*len(y_test_sr)
	return X_test_sr, y_test_sr, docs_test_sr



if __name__ == "__main__":
	# Config
	import config
	input_dir = config.input_dir
	output_dir = config.output_dir
	model = config.model
	run_version_number = config.run_version_number
	subreddits = config.subreddits
	cv = int(config.cv)
	subsample = int(config.subsample)
	include_subreddits_overN = int(config.include_subreddits_overN)
	run_modelN = int(config.run_modelN)
	run_final_model = config.run_final_model
	dim_reduction = config.dim_reduction
	task = config.task
	midpandemic_train = config.midpandemic_train
	midpandemic_test = config.midpandemic_test
	subsample_midpandemic_test =  config.subsample_midpandemic_test
	pre_or_post = config.pre_or_post
	timestep = config.timestep

	# Load data
	# ===========================================================================
	subreddit = subreddits[config.subredditN]
	if run_final_model:
		output_dir = data_helpers.make_output_dir(output_dir, name='run_final_model_v{}_model{}_{}'.format(run_version_number, run_modelN, subreddit))
	else:
		output_dir = data_helpers.make_output_dir(output_dir, name='run_gridsearch_v{}_model{}_{}'.format(run_version_number, run_modelN, subreddit))

	if task == 'binary':
		reddit_data = load_reddit.binary(input_dir, subreddit, subreddits,pre_or_post=pre_or_post, subsample=subsample )

		# Create additional test sets, 1 for same subreddit but of midpandemic data and 1 for COVID19
		if midpandemic_test:
			subreddits_midpandemic = subreddits+['COVID19_support']
			midpandemic_test_data = load_reddit.multiclass(input_dir, subreddits_midpandemic ,
			                            pre_or_post='post',subsample=subsample_midpandemic_test)


	elif task == 'multiclass':
		reddit_data = load_reddit.multiclass(input_dir, subreddits, pre_or_post = 'pre')



	# Convert df to X,y
	X_train, X_test, y_train, y_test, docs_train, docs_test, features = df_to_X(reddit_data, task)


	if midpandemic_test:
		X_test_covid, y_test_covid, docs_test_covid= df_to_X_midpandemic(midpandemic_test_data , timestep=None,
		                                                          filter_days=['2020/03/11', '2020/04/20'],
		                                                          subreddit='COVID19_support')
		X_test_sr, y_test_sr, docs_test_sr = df_to_X_midpandemic(midpandemic_test_data , timestep=None,
		                                                          filter_days=['2020/03/11', '2020/04/20'],
		                                                          subreddit=subreddit)


	print('===loaded data====')

	# Count
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

	# Extract tfidf
	train_tfidf, test_tfidf, feature_names_tfidf = extract_features.tfidf(X_train_sentences=docs_train, X_test_sentences=docs_test,
	                                                                      ngram_range=(1, 2),
	                                                                      max_features=256, min_df=2, max_df=0.8,
	                                                                      model=model, stem=config.stem)
	X_train = np.concatenate([X_train, train_tfidf], axis=1)
	X_test = np.concatenate([X_test, test_tfidf], axis=1)
	features = np.concatenate([features, feature_names_tfidf], axis=0)


	if midpandemic_test:
		train_tfidf, test_tfidf, feature_names_tfidf = extract_features.tfidf(X_train_sentences=docs_train,
		                                                                      X_test_sentences=docs_test_covid,
		                                                                      ngram_range=(1, 2),
		                                                                      max_features=256, min_df=2, max_df=0.8,
		                                                                      model=model, stem=config.stem)
		X_test_covid = np.concatenate([X_test_covid, test_tfidf], axis=1)

		train_tfidf, test_tfidf, feature_names_tfidf = extract_features.tfidf(X_train_sentences=docs_train,
		                                                                      X_test_sentences=docs_test_sr,
		                                                                      ngram_range=(1, 2),
		                                                                      max_features=256, min_df=2, max_df=0.8,
		                                                                      model=model, stem=config.stem)
		X_test_sr = np.concatenate([X_test_sr, test_tfidf], axis=1)

	# Run models
	# ================================================================================
	subreddits = list(np.unique(reddit_data.subreddit))
	if task == 'binary':
		subreddits = ['control', subreddit]

	if run_final_model:
		parameters = config_parameters.final_pipeline(run_modelN)
	else:
		# gridsearch
		parameters = config_parameters.parameters_all_models(y_train, dim_reduction=dim_reduction)

	# write all variables in config)
	with open(output_dir + 'config.txt', 'a+') as f:
		f.write(str(subreddits))
		f.write('\n')
		f.write(str(parameters))
		f.write('\n')

	if run_final_model:
		final_model(X_train, y_train, X_test, y_test,run_modelN, subreddit, subreddits,features,output_dir)

		if midpandemic_test:
			# Here we dont care about labels, only y_probs
			final_model(X_train, y_train, X_test_covid, y_test_covid, run_modelN, subreddit, subreddits, features,
			            output_dir, append_to_name = 'covid19')

			final_model(X_train, y_train, X_test_sr, y_test_sr, run_modelN, subreddit, subreddits, features,
			            output_dir,append_to_name = 'midpandemic')

	else:
		# Hyperparameter tuning
		# ========================================================================
		if dim_reduction:
			pipeline = Pipeline([
				('normalization', None),
				('umap', umap.UMAP(n_components=2, min_dist=0.1, metric='correlation', random_state=seed_value)),
				('clf', switcher.ClfSwitcher()),
			])
		else:
			pipeline = Pipeline([
				('normalization', None),
				('feature_selection', SelectKBest()),
				('clf', switcher.ClfSwitcher()),
			])

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
			print('=====================\n')
			model_name= str(results.param_clf__estimator[0]).split('(')[0]
			model_name = model_name.replace('[','')

			with open(output_dir+model_name+'.txt', 'a+') as f:
				f.write('\n=======================================================\n')
				f.write(str(gscv.best_estimator_))
				f.write('\n')
				f.write(str(np.round(gscv.best_score_,4)))
				f.write('\n=======================================================\n')

			results.to_csv(output_dir+model_name+'.csv',index_label=0)

