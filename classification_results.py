#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''

import pandas as pd
import numpy as np
import os
pd.options.display.width = 0







if __name__ == "__main__":



	input_dir = './../../datum/reddit/output/binary5b/'
	output_dir = './../../datum/reddit/output/binary5b/'
	dirs = os.listdir(input_dir)
	# dirs = [n for n in dirs if '20-05-08-06-48' in n]
	results = []
	coefs_all = []
	for dir in dirs:
		if dir in ['.DS_Store', 'summary.txt', 'summary.csv', 'summary_results.csv','summary_coefs.csv']:
			continue
		subreddit = dir.split('_')[-1]
		result = np.round(pd.read_csv(input_dir+dir+'/report_SGDClassifier.csv')['f1-score'][4],2)
		coefs = pd.read_csv(input_dir+dir+'/coefs_df_SGDClassifier.csv', index_col=0).sort_values(subreddit)[-6:]
		coefs = coefs.round(1)
		coefs.columns = ['coefficients']
		coefs['subreddit'] = [subreddit]*coefs.shape[0]
		# coefs = coefs.reset_index()
		results.append([subreddit, result])
		coefs_all.append(coefs)
		with open(input_dir + 'summary.txt', 'a+') as f:
			f.write('\n')
			f.write(str(coefs))
			f.write('\n')

	results = pd.DataFrame(results)
	results.columns = ['subreddit', 'Weighted F1']
	results = results.sort_values('subreddit')
	results.to_csv(input_dir+'summary_results.csv')
	with open(input_dir+'summary.txt', 'a+') as f:
		f.write('\n')
		f.write(results.to_latex())
		f.write('\n')

	coefs_all2= pd.concat(coefs_all,axis=0)
	coefs_all2.to_csv(input_dir + 'summary_coefs.csv')







	subreddits = ['addiction', 'EDAnonymous', 'adhd', 'autism', 'alcoholism', 'bipolarreddit', 'depression', 'anxiety',
	              'healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']




	subreddits_irr = ['EDAnonymous', 'alcoholism', 'bipolarreddit', 'depression', 'anxiety','schizophrenia', ]
	results_irr = results[results.subreddit.isin(subreddits_irr)]
	irr =           [0.56, 0.4,0.4,0.28,0.2,0.46,] #https://ajp.psychiatryonline.org/doi/full/10.1176/appi.ajp.2012.12091189
	results_irr2 =  [0.91,0.93,0.8,0.79,0.86,0.81]
	results_irr2 = [0.91, 0.93, 0.8, 0.79, 0.81, 0.81]
	from scipy import stats
	r,p = stats.spearmanr(irr, results_irr2 )
	print(r,p)
	import matplotlib.pyplot as plt
	plt.plot(irr,label = 'inter-reliability rates')
	plt.plot(results_irr2,label = 'performance')
	plt.text(y = 0.5, x = 0.5,
	         s = 'r: {} ({})'.format(np.round(r,2), np.round(p,2)))
	plt.xticks(range(len(irr)),labels = subreddits_irr, rotation='vertical')
	plt.tight_layout()
	plt.legend()




