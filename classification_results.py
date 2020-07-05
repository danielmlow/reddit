#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''

from subprocess import Popen
import json
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


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
			coefs.columns = ['coefficients']
			coefs['subreddit'] = [subreddit] * coefs.shape[0]
			coefs_all.append(coefs)


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
	df = df.rename(columns={'predicted': 'Predicted from\nCOVID19_support'})

	df.set_index('subreddit', inplace=True)

	# csv
	df.to_csv(output_dir + 'psych_profiler{}.csv'.format(test_set))
	# heatmap

	if plot:
		plt.figure(figsize=(3,7))
		hm = sns.heatmap(df.iloc[:,:1], annot=True, linewidths=0.1, cbar=False, square=True, vmin=0, vmax=0.6)
		bottom, top = hm.get_ylim()
		hm.set_ylim(bottom + 0.5, top - 0.5)
		plt.ylabel('Binary classifier')
		plt.tight_layout()
		plt.savefig(output_dir + 'psych_profiler{}.png'.format(test_set), dpi=300)

	# latex
	mean_probs_all_latex= df.iloc[:,:1].to_latex(index=True)
	with open(output_dir + 'psych_profiler{}.txt'.format(test_set), 'a+') as f:
		f.write(mean_probs_all_latex)





if __name__ == "__main__":

	# Change path and model used for subsequent analyses. binary8 is where I put all results, 8 means 8th version of dataset
	version = 8
	input_dir = f'./../../datum/reddit/output/classification/binary{version}/'
	# Change path for chosen model (SGD L1 in our case) for subsequent analysis
	chosen_model = 0
	input_dir_1model = f'./../../datum/reddit/output/classification/binary{version}_model{chosen_model}/'
	try: os.mkdir(input_dir_1model)
	except: pass
	

	# Run
	models = range(0,5)
	# These are how the files are named automatically from run.py
	model_names = {0:'SGDClassifier',
				   1:'SGDClassifier',
				   2: 'SVC',
				   3: 'ExtraTreesClassifier',
				   4: 'XGBModel',
				   }
	# How models will apear in table, you can change
	model_names_publication= {0:'SGD L1',
				   1:'SGD EN',
				   2: 'SVM',
				   3: 'ET',
				   4: 'XGB',
				   }

	for model in models:
		model_name = model_names.get(model)
		results_pre =  summarize(input_dir, test_set='', model=model, model_name=model_name)
		# results_mid = summarize(input_dir, test_set='_midpandemic', model=model, model_name=model_name)
		# results_covid = summarize(input_dir, test_set='_covid19', model=model, model_name=model_name)


	# Count proportion of nonzero features
	total_possible_coefs = 15*346 # 14 binary classifiers, 346 features (256 are tfidf)
	nonzero_coefs_all = []
	for model in models[:3]: #last two did not have coefs computed because they're tree ensemble based models
		nonzero_coefs = pd.read_csv(input_dir+f'summary_model{model}/summary_coefs.csv', index_col=0).shape[0]
		nonzero_coefs_all.append(f'{nonzero_coefs} ({int(nonzero_coefs/total_possible_coefs*100)})')

	for model in models[3:]:
		nonzero_coefs_all.append(f'{total_possible_coefs} ({100})')


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



	# Mean
	results_mean = pd.DataFrame(results_all_models.mean()).T
	results_mean['subreddit'] = 'Mean'
	cols = results_mean.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	results_mean= results_mean[cols]
	results_mean=results_mean.round(3)
	results_all_models = results_all_models.append(results_mean, ignore_index=True)

	# Nonzero coefs
	results_coefs = pd.DataFrame(nonzero_coefs_all).T
	results_coefs['subreddit'] = 'Model complexity No. (\%)'
	cols = results_coefs.columns.tolist()
	cols = cols[-1:] + cols[:-1]

	results_coefs = results_coefs[cols]
	results_coefs.columns = results_all_models.columns
	results_all_models = results_all_models.append(results_coefs, ignore_index=True)

	# Format
	results_all_models.set_index('subreddit', inplace=True)
	results_all_models = results_all_models.round(3)
	# Save
	results_all_models.to_csv(input_dir+'results_all_models.csv')
	results_all_models_latex= results_all_models.to_latex(index=True)
	with open(input_dir+ 'results_all_models_latex.txt', 'a+') as f:
		f.write(results_all_models_latex)





	# This will only be done for the chosen model
	# ==================================================================================================================
	model = chosen_model
	model_name = model_names.get(model)

	# copy results for model to different directory to add more results in a more tidy way
	Popen(str('scp -r ' + input_dir+f'*model{chosen_model}* '+input_dir_1model), shell=True)

	results_pre =  summarize(input_dir_1model, test_set='', model=model, model_name=model_name)
	results_mid = summarize(input_dir_1model, test_set='_midpandemic', model=model, model_name=model_name)
	results_covid = summarize(input_dir_1model, test_set='_covid19', model=model, model_name=model_name)

	# Obtain sizes of additional test sets
	dirs = os.listdir(input_dir_1model)
	dirs = [n for n in dirs if 'run_final' in n]
	midpandemic = []
	covid19 = []
	for d in dirs:
		df = pd.read_csv(input_dir_1model+d+'/report_SGDClassifier_midpandemic.csv', index_col=0)
		midpandemic.append(df.support[-1])
		df = pd.read_csv(input_dir_1model + d + '/report_SGDClassifier_covid19.csv', index_col=0)
		covid19.append(df.support[-1])

	print(f'midpandemic: {np.round(np.mean(midpandemic),2)} ({np.round(np.std(midpandemic),2)})')
	print(f'covid19: {np.round(np.mean(covid19), 2)} ({np.round(np.std(covid19), 5)})')


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



	results.to_csv(input_dir_1model+'summary_model{}/results_pre_vs_mid.csv'.format(model))
	results_latex = results.to_latex(index=True)
	with open(input_dir_1model+'summary_model{}/results_pre_vs_mid_latex.txt'.format(model), 'a+') as f:
		f.write(results_latex )
		f.write('\n')


	# Psych profiler
	psych_profiler(input_dir_1model, test_set='_covid19', model = model,model_name = model_names.get(model), plot=True)




	# POSITIVE Coef samples
	n_features = 8 #how many top coefs
	# These are all coefs for all models stacked.
	coefs = pd.read_csv(input_dir_1model + 'summary_model{}/summary_coefs.csv'.format(model), index_col='Unnamed: 0')
	subreddits=np.unique(coefs.subreddit)
	coefs_top = []
	for sr in subreddits:
		coefs_sr = coefs[coefs.subreddit==sr]
		coefs_sr = coefs_sr .loc[~coefs_sr .index.duplicated(keep='last')] # TFIDF sometimes created duplicate feature
		features_sr =list(coefs_sr.index)[-n_features:][::-1]
		features_sr = [n.replace('tfidf_','').replace('liwc_', 'LIWC ').replace('n ', 'N ').replace(
			'sent_neu', 'neutral sentiment').replace('sent_compound', 'compound sentiment').replace(
			'sent_pos', 'positive sentiment').replace('sent_neg','negative sentiment').replace(
			'_total', ' lexicon').replace('_', ' ').replace('oN', 'on') for n in features_sr ]

		coefs_top.append(['r/'+sr, ', '.join(features_sr)])
		# split into two rows if using Latex
		# features_sr_0 = str(features_sr[:7] ).replace('[','').replace(']','').replace("'","")
		# features_sr_1 = str(features_sr[7:]).replace('[', '').replace(']', '').replace("'", "")
		# coefs_top.append([sr,features_sr_0])
		# coefs_top.append(['',features_sr_1])

	coefs_top_pos = pd.DataFrame(coefs_top)
	coefs_top_pos.columns = ['Subreddit', f'Top {n_features} important features - positive coefficients']
	coefs_top_pos.to_csv(input_dir_1model+f'top_{n_features}_important_features_positive.csv')



	# NEGATIVE Coef samples
	coefs = pd.read_csv(input_dir_1model + 'summary_model{}/summary_coefs.csv'.format(model), index_col='Unnamed: 0')

	subreddits=np.unique(coefs.subreddit)
	coefs_top = []
	dirs = os.listdir(input_dir_1model)
	for sr in subreddits:
		dir_sr = [n for n in dirs if '_'+sr in n][0]
		coefs = pd.read_csv(input_dir_1model + dir_sr+f'/coefs_df_{model_name}.csv', index_col='Unnamed: 0')

		coefs_sr = coefs.sort_values(sr)
		coefs_sr = coefs_sr.loc[~coefs_sr.index.duplicated(keep='first')]  # TFIDF sometimes created duplicate feature
		coefs_sr = coefs_sr[:n_features]
		features_sr =coefs_sr.index
		features_sr = [n.replace('tfidf_', '').replace('liwc_', 'LIWC ').replace('n ', 'N ').replace(
			'sent_neu', 'neutral sentiment').replace('sent_compound', 'compound sentiment').replace(
			'sent_pos', 'positive sentiment').replace('sent_neg', 'negative sentiment').replace(
			'_total', ' lexicon').replace('_', ' ').replace('oN', 'on') for n in features_sr]
		coefs_top.append(['r/'+sr, ', '.join(features_sr)])
		# features_sr_0 = str(features_sr[:7] ).replace('[','').replace(']','').replace("'","")
		# features_sr_1 = str(features_sr[7:]).replace('[', '').replace(']', '').replace("'", "")
		# coefs_top.append([sr,features_sr_0])
		# coefs_top.append(['',features_sr_1])

	coefs_top_neg = pd.DataFrame(coefs_top)
	coefs_top_neg.columns = ['Subreddit', f'Top {n_features} important features - negative coefficients']
	coefs_top_neg.to_csv(input_dir_1model+f'top_{n_features}_important_features_negative.csv')

	# combined
	coefs_top = coefs_top_pos.merge(coefs_top_neg)
	coefs_top.to_csv(input_dir_1model + f'top_{n_features}_important_features_all.csv')




	# Max coefficients
	# ====

	coefs = pd.read_csv(input_dir_1model+'summary_model{}/summary_coefs.csv'.format(model), index_col='Unnamed: 0')
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
	max_all.to_csv(input_dir_1model+'main_sr_per_feature.csv', index=False)






	with open(input_dir_1model+'main_sr_per_feature.json', 'w') as fp:
		json.dump(max_d, fp)

