#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''

import pandas as pd
import numpy as np
import os
pd.options.display.width = 0

def summarize(input_dir, test_set=''):
	output_dir = input_dir + '/summary/'
	try:
		os.mkdir(output_dir )
	except: pass
	dirs = os.listdir(input_dir)
	results = []
	coefs_all = []
	for dir in dirs:
		if dir in ['.DS_Store', 'summary']:
			continue
		subreddit = dir.split('_')[-1]
		result = np.round(pd.read_csv(input_dir + dir + '/report_SGDClassifier{}.csv'.format(test_set))['f1-score'][4], 2)
		coefs = pd.read_csv(input_dir + dir + '/coefs_df_SGDClassifier{}.csv'.format(test_set), index_col=0).sort_values(subreddit)
		# Select only positive
		coefs = coefs[coefs[subreddit]>0]
		# coefs = coefs.round(1)
		coefs.columns = ['coefficients']
		coefs['subreddit'] = [subreddit] * coefs.shape[0]
		# coefs = coefs.reset_index()
		results.append([subreddit, result])
		coefs_all.append(coefs)
		with open(output_dir+ 'summary{}.txt'.format(test_set), 'a+') as f:
			f.write('\n')
			f.write(str(coefs))
			f.write('\n')

	results = pd.DataFrame(results)
	results.columns = ['subreddit', 'Weighted F1']
	results = results.sort_values('subreddit')
	results.to_csv(output_dir+ 'summary_results{}.csv'.format(test_set))
	with open(output_dir+ 'summary{}.txt'.format(test_set), 'a+') as f:
		f.write('\n')
		f.write(results.to_latex())
		f.write('\n')

	coefs_all2 = pd.concat(coefs_all, axis=0)
	coefs_all2.to_csv(output_dir+ 'summary_coefs{}.csv'.format(test_set))

	return results



if __name__ == "__main__":



	input_dir = './../../datum/reddit/output/binary6/'
	output_dir = './../../datum/reddit/output/binary6/'
	results_pre =  summarize(input_dir, test_set='')
	results_mid = summarize(input_dir, test_set='_midpandemic')
	results_covid = summarize(input_dir, test_set='_covid19')

	# show pre vs mid pandemic test results side by side
	results_pre = results_pre.rename(columns={'Weighted F1':'F1 pre'})
	results_mid = results_mid.rename(columns={'Weighted F1': 'F1 mid'})
	results = results_pre.merge(results_mid, on='subreddit')
	results.set_index('subreddit', inplace=True)
	results.to_csv(output_dir+'summary/results_pre_vs_mid.csv')
	results_latex = results.to_latex(index=True)
	with open(output_dir+'summary/results_pre_vs_mid_latex.txt', 'a+') as f:
		f.write(results_latex )



	#
	#
	#
	# subreddits = ['addiction', 'EDAnonymous', 'adhd', 'autism', 'alcoholism', 'bipolarreddit', 'depression', 'anxiety',
	#               'healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']
	#
	#
	#
	#
	# subreddits_irr = ['EDAnonymous', 'alcoholism', 'bipolarreddit', 'depression', 'anxiety','schizophrenia', ]
	# results_irr = results[results.subreddit.isin(subreddits_irr)]
	# irr =           [0.56, 0.4,0.4,0.28,0.2,0.46,] #https://ajp.psychiatryonline.org/doi/full/10.1176/appi.ajp.2012.12091189
	# results_irr2 =  [0.91,0.93,0.8,0.79,0.86,0.81]
	# results_irr2 = [0.91, 0.93, 0.8, 0.79, 0.81, 0.81]
	# from scipy import stats
	# r,p = stats.spearmanr(irr, results_irr2 )
	# print(r,p)
	# import matplotlib.pyplot as plt
	# plt.plot(irr,label = 'inter-reliability rates')
	# plt.plot(results_irr2,label = 'performance')
	# plt.text(y = 0.5, x = 0.5,
	#          s = 'r: {} ({})'.format(np.round(r,2), np.round(p,2)))
	# plt.xticks(range(len(irr)),labels = subreddits_irr, rotation='vertical')
	# plt.tight_layout()
	# plt.legend()
	#
	#
	#
	#
