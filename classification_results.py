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



	input_dir = './../../datum/reddit/output/binary/'
	output_dir = './../../datum/reddit/output/binary/'
	dirs = os.listdir(input_dir)
	coefs_all = []
	results = []
	try: dirs.remove('.DS_Store')
	except: pass
	for dir in dirs:
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
	results.to_csv(input_dir+'summary_results.csv')

	coefs_all2= pd.concat(coefs_all,axis=0)
	coefs_all2.to_csv(input_dir + 'summary_coefs.csv')
	with open(input_dir + 'summary.txt', 'a+') as f:
			f.write(str(coefs_all2[:]))




	with open(input_dir+'summary.txt', 'a+') as f:
		f.write(results.to_latex())
		f.write('\n')








	subreddits = ['addiction', 'EDAnonymous', 'adhd', 'autism', 'alcoholism', 'bipolarreddit', 'depression', 'anxiety',
	              'healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']



