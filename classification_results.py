#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

pd.options.display.width = 0

def summarize(input_dir, test_set='', model=0, model_name='SGDClassifier'):
	output_dir = input_dir + 'summary_model{}/'.format(model)

	try:
		os.mkdir(output_dir )
	except: pass
	dirs = os.listdir(input_dir)
	dirs = [n for n in dirs if 'model{}'.format(model) in n]

	results = []
	coefs_all = []
	for dir in dirs:
		if '.DS_Store' in dir or 'summary' in dir:
			continue
		subreddit = dir.split('_')[-1]
		result = np.round(pd.read_csv(input_dir + dir + '/report_{}{}.csv'.format(model_name, test_set))['f1-score'][4], 3)
		results.append([subreddit, result])

		if model_name in ['SGDClassifier', 'SVC']:
			coefs = pd.read_csv(input_dir + dir + '/coefs_df_{}{}.csv'.format(model_name, test_set),
			                    index_col=0).sort_values(subreddit)
			# Select only positive

			coefs = coefs[coefs[subreddit]>0]
			# coefs = coefs.round(1)
			coefs.columns = ['coefficients']
			coefs['subreddit'] = [subreddit] * coefs.shape[0]
			coefs_all.append(coefs)
			# coefs = coefs.reset_index()


			with open(output_dir+ 'summary{}.txt'.format(test_set), 'a+') as f:
				f.write('\n')
				f.write(str(coefs))
				f.write('\n')


	if model_name in ['SGDClassifier', 'SVC']:
		coefs_all2 = pd.concat(coefs_all, axis=0)
		coefs_all2.to_csv(output_dir + 'summary_coefs{}.csv'.format(test_set))

	results = pd.DataFrame(results)
	results.columns = ['subreddit', 'Weighted F1']
	results = results.sort_values('subreddit')
	results.to_csv(output_dir+ 'summary_results{}.csv'.format(test_set))
	with open(output_dir+ 'summary{}.txt'.format(test_set), 'a+') as f:
		f.write('\n')
		f.write(results.to_latex())
		f.write('\n')



	return results


def psych_profiler(input_dir, test_set='_covid19', model=0,model_name='SGDClassifier', plot=False):
	output_dir = input_dir + 'summary_model{}/'.format(model)
	try:
		os.mkdir(output_dir)
	except:
		pass
	dirs = os.listdir(input_dir)
	proportion_classified_as_sr = []

	for dir in dirs:
		if dir in ['.DS_Store', 'summary'] or '.' in dir or 'summary' in dir:
			continue
		subreddit = dir.split('_')[-1]

		# 'coefs_df_SGDClassifier_covid19.csv'

		y_pred_probs = pd.read_csv(input_dir + dir + '/y_pred_probs_{}{}.csv'.format(model_name,test_set))

		# Take the mean prob for this subreddit
		# mean_probs = y_pred_probs.mean()[subreddit]
		# mean_probs_all.append([subreddit, mean_probs])

		# What percent of posts are classified as subreddit?
		y_pred_probs_sr = y_pred_probs[subreddit].values
		# Here I use 0.5 cutoff but other could be used.
		y_pred_proportion = np.sum(np.round(y_pred_probs_sr))/len(y_pred_probs_sr)
		proportion_classified_as_sr.append([subreddit, y_pred_proportion, y_pred_probs_sr])

	df = pd.DataFrame(proportion_classified_as_sr).round(2)

	df.columns = ['subreddit', 'predicted', 'y_pred_probs_sr']
	df = df.sort_values('predicted')[::-1]
	df.set_index('subreddit', inplace=True)
	# csv
	df.to_csv(output_dir + 'psych_profiler{}.csv'.format(test_set))
	# heatmap

	if plot:
		plt.figure(figsize=(3,7))
		sns.heatmap(df.iloc[:,:1], annot=True, linewidths=0.1, cbar=False)
		plt.tight_layout()
		plt.savefig(output_dir + 'psych_profiler{}.png'.format(test_set), dpi=300)

	# latex
	mean_probs_all_latex= df.iloc[:,:1].to_latex(index=True)
	with open(output_dir + 'psych_profiler{}.txt'.format(test_set), 'a+') as f:
		f.write(mean_probs_all_latex)





if __name__ == "__main__":


	# Change path and model used for subsequent analyses
	input_dir = './../../datum/reddit/output/binary7_all_models/'
	highest_performing_model = 2

	# Run
	models = range(0,5)
	model_names = {0:'SGDClassifier',
	               1:'SGDClassifier',
	               2: 'SVC',
	               3: 'ExtraTreesClassifier',
	               4: 'XGBModel',
	               }

	model_names_publication= {0:'L1',
	               1:'EN',
	               2: 'SVM',
	               3: 'ET',
	               4: 'XGB',
	               }



	for model in models:
		model_name = model_names.get(model)
		results_pre =  summarize(input_dir, test_set='', model=model, model_name=model_name)
		# results_mid = summarize(input_dir, test_set='_midpandemic', model=model, model_name=model_name)
		# results_covid = summarize(input_dir, test_set='_covid19', model=model, model_name=model_name)


	# Merge all model results
	dirs= os.listdir(input_dir)
	dirs = [n for n in dirs if 'summary' in n]

	model = 0
	model_name = model_names_publication.get(model)
	results_all_models = pd.read_csv(input_dir + 'summary_model{}/summary_results.csv'.format(0), index_col='Unnamed: 0')
	results_all_models.columns = ['subreddit', '{}'.format(model_name )]

	for model in models[1:]:
		model_name = model_names_publication.get(model)
		results = pd.read_csv(input_dir + 'summary_model{}/summary_results.csv'.format(model), index_col='Unnamed: 0')
		results.columns = ['subreddit', '{}'.format(model_name )]
		results_all_models = results_all_models.merge(results, on='subreddit')

	results_mean = pd.DataFrame(results_all_models.mean()).T
	results_mean['subreddit'] = 'Mean'
	cols = results_mean.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	results_mean= results_mean[cols]
	results_all_models = results_all_models.append(results_mean, ignore_index=True)
	results_all_models.set_index('subreddit', inplace=True)
	results_all_models = results_all_models.round(3)
	results_all_models.to_csv(input_dir+'results_all_models.csv')
	results_all_models_latex= results_all_models.to_latex(index=True)
	with open(input_dir+ 'results_all_models_latex.txt', 'a+') as f:
		f.write(results_all_models_latex)



	# Choose model: 0,1,2,3, or 4
	input_dir = f'./../../datum/reddit/output/binary7_model{highest_performing_model}/'
	for model in [highest_performing_model]:
		model_name = model_names.get(model)
		results_pre =  summarize(input_dir, test_set='', model=model, model_name=model_name)
		results_mid = summarize(input_dir, test_set='_midpandemic', model=model, model_name=model_name)
		results_covid = summarize(input_dir, test_set='_covid19', model=model, model_name=model_name)



	# show pre vs mid pandemic test results side by side
	results_pre = results_pre.rename(columns={'Weighted F1':'F1 pre'})
	results_mid = results_mid.rename(columns={'Weighted F1': 'F1 mid'})
	results = results_pre.merge(results_mid, on='subreddit')
	# add delta col
	delta = results.iloc[:,1]-results.iloc[:,2]
	delta = delta.abs()
	results['Change'] = delta
	results = results.sort_values('Change')[::-1]
	# Add mean
	results_mean = pd.DataFrame(results.mean()).T
	results_mean['subreddit'] = ['Mean']
	cols = results_mean.columns.tolist()
	results= results.append(results_mean, ignore_index=True)
	results.set_index('subreddit', inplace=True)
	results= results[['F1 pre', 'F1 mid', 'Change']]
	results = results.round(3)



	stat, p = stats.ttest_ind(results.iloc[:, 0], results.iloc[:, 1])
	# We hypothesize that accuracy will decrease
	one_sided_p = p / 2



	results.to_csv(input_dir+'summary_model{}/results_pre_vs_mid.csv'.format(model))
	results_latex = results.to_latex(index=True)
	with open(input_dir+'summary_model{}/results_pre_vs_mid_latex.txt'.format(model), 'a+') as f:
		f.write(results_latex )
		f.write('\n')
		f.write('t-test ({}) p-value={}'.format(stat, one_sided_p))

	# Psych profiler
	psych_profiler(input_dir, test_set='_covid19', model = model,model_name = model_names.get(model), plot=True)
	# ax1 = sns.heatmap(hmap, cbar=0, cmap="YlGnBu", linewidths=2, ax=ax0, vmax=3000, vmin=0, square=True)



	# POSITIVE Coef samples
	coefs = pd.read_csv(input_dir + 'summary_model{}/summary_coefs.csv'.format(model), index_col='Unnamed: 0')
	subreddits=np.unique(coefs.subreddit)
	coefs_top = []
	for sr in subreddits:
		coefs_sr = coefs[coefs.subreddit==sr][-14:]
		feaures_sr =coefs_sr.index
		feaures_sr = [n.replace('tfidf_','') for n in feaures_sr ][::-1]
		feaures_sr_0 = str(feaures_sr[:7] ).replace('[','').replace(']','').replace("'","")
		feaures_sr_1 = str(feaures_sr[7:]).replace('[', '').replace(']', '').replace("'", "")
		coefs_top.append([sr,feaures_sr_0])
		coefs_top.append(['',feaures_sr_1])


	coefs_top = pd.DataFrame(coefs_top)
	coefs_top.columns = ['Subreddit', 'Top 10 important features - positive coefficients']
	coefs_top.to_csv(input_dir+'top_10_important_features_positive.csv')


	# coefs_top_latex= coefs_top.to_latex(index=True,)
	# with open(input_dir + 'top_10_important_features.tex', 'a+') as f:
	# 	f.write(coefs_top_latex)
	# with pd.option_context("max_colwidth", 5000):
	# 	print(coefs_top_latex)


	# NEGATIVE Coef samples
	coefs = pd.read_csv(input_dir + 'summary_model{}/summary_coefs.csv'.format(model), index_col='Unnamed: 0')

	subreddits=np.unique(coefs.subreddit)
	coefs_top = []
	dirs = os.listdir(input_dir)
	for sr in subreddits:
		dir_sr = [n for n in dirs if '_'+sr in n][0]
		coefs = pd.read_csv(input_dir + dir_sr+f'/coefs_df_{model_name}.csv', index_col='Unnamed: 0')

		coefs_sr = coefs.sort_values(sr)[:14]
		feaures_sr =coefs_sr.index
		feaures_sr = [n.replace('tfidf_','') for n in feaures_sr ]
		feaures_sr_0 = str(feaures_sr[:7] ).replace('[','').replace(']','').replace("'","")
		feaures_sr_1 = str(feaures_sr[7:]).replace('[', '').replace(']', '').replace("'", "")
		coefs_top.append([sr,feaures_sr_0])
		coefs_top.append(['',feaures_sr_1])

	coefs_top = pd.DataFrame(coefs_top)
	coefs_top.columns = ['Subreddit', 'Top 10 important features - negative coefficients']
	coefs_top.to_csv(input_dir+'top_10_important_features_negative.csv')


	# Max coefficients
	# ====

	coefs = pd.read_csv(input_dir+'summary_model{}/summary_coefs.csv'.format(model), index_col='Unnamed: 0')
	features = np.unique(coefs.index)
	print('n features: ', len(features))


	percentile_coef = np.percentile(coefs.coefficients, 50)

	features_high = np.unique(coefs[coefs.coefficients>percentile_coef].index)
	print('n features: ', len(features_high ))
	no_tfidf = [n for n in features_high if 'tfidf' not in n]
	print('n features: ', len(no_tfidf ))

	max_all = []
	max_d = {}
	for feature in no_tfidf:
		df_feature = coefs[coefs.index==feature]
		print(df_feature )
		print('=====\n')
		df_feature_max = [df_feature.max().subreddit, df_feature.index[0]]
		highest_subreddit = df_feature.sort_values('coefficients')['subreddit'].tolist()[-1]
		max_d[feature]=highest_subreddit
		print(feature,highest_subreddit)
		max_all.append(df_feature_max)

	max_all = pd.DataFrame(max_all)
	max_all.to_csv(input_dir+'main_sr_per_feature.csv', index=False)

	import json




	with open(input_dir+'main_sr_per_feature.json', 'w') as fp:
		json.dump(max_d, fp)

