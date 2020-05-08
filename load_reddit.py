
import numpy as np
import pandas as pd

def subsample_df(df, subsample):
	if type(subsample) == float:
		subsample = int(df.shape[0]*subsample)
	df = df.reset_index(drop=True)
	df2 = df.loc[np.random.choice(df.index,subsample, replace=False)]
	return df2



def multiclass(input_dir, subreddits, pre_or_post = 'pre', subsample=None, subsample_controls=None, subsample_subreddits_overN=None, days = (0,-1)):

	# Careful: if you add COVID19_support and it does not exist in the first time step, then this will confuse metric learning
	if pre_or_post == 'post':
	# 	collect mid pandemic data
		# todo
		pass

	if pre_or_post == 'pre':
		reddit_data = pd.read_csv(input_dir +subreddits[0]+'_{}_features.csv'.format(pre_or_post), index_col=False)
		print('before:', subreddits[0], reddit_data.shape)
		if subsample_controls:
			reddit_data = subsample_df(reddit_data, subsample_controls)
			print('after:', subreddits[0], reddit_data.shape)
		elif subsample:
			reddit_data = subsample_df(reddit_data, subsample)
			print('after:', subreddits[0], reddit_data.shape)

		for i in np.arange(1, len(subreddits)):
			new_data = pd.read_csv(input_dir+subreddits[i]+'_{}_features.csv'.format(pre_or_post))
			print('before:',subreddits[i], new_data.shape)
			if subsample_controls:
				new_data = subsample_df(new_data, subsample_controls)
			elif subsample:
				new_data = subsample_df(new_data, subsample)

			print('after:',subreddits[i], new_data.shape)
			reddit_data = pd.concat([reddit_data, new_data], axis=0)

	return reddit_data



def binary(input_dir, subreddit, control_subreddits, pre_or_post = 'pre', subsample=None, subsample_controls=None, subsample_subreddits_overN=None):
	# Careful: if you add COVID19_support and it does not exist in the first time step, then this will confuse metric learning
	if pre_or_post == 'post':
		# collect mid-pandemic data
		reddit_data = multiclass(input_dir, control_subreddits, pre_or_post=pre_or_post, subsample=subsample,
		                         subsample_subreddits_overN=None,
		                         days=(0, -1))


	if pre_or_post == 'pre':
		# collect pre-pandemic data
		reddit_data = multiclass(input_dir, control_subreddits, pre_or_post=pre_or_post,
		                         subsample=subsample, subsample_controls=subsample_controls ,subsample_subreddits_overN=None,
		                       days=(0, -1))
		reddit_data_sr = pd.read_csv(input_dir + subreddit + '_{}_features.csv'.format(pre_or_post), index_col=False)
		# reddit_data_sr = reddit_data[reddit_data.subreddit == subreddit]
		reddit_data_sr = subsample_df(reddit_data_sr , subsample)

		# Exclude certain the same or overlapping subreddits from control group
		if subreddit == 'healthanxiety':
			exclude = ['healthanxiety','anxiety']
		elif subreddit == 'socialanxiety':
			exclude = ['socialanxiety', 'anxiety']
		elif subreddit == 'anxiety':
			exclude = ['anxiety','healthanxiety','socialanxiety']
		elif subreddit == 'depression':
			exclude = ['depression','suicidewatch']
		elif subreddit == 'suicidewatch':
			exclude = ['suicidewatch','depression']
		# elif subreddit == 'bipolarreddit':
		# 	exclude = ['bipolarreddit'] #schizophrenia 0.14, healthanxiety 0.08
		# elif subreddit == 'schizophrenia':
		# 	exclude = ['schizophrenia'] #bipolar 0.16, healthanxiety 0.10
		else:
			exclude = [subreddit] #socialanxiety 0.11


		# create control df: remove the treatment and overlapping subreddits
		reddit_data_controls = reddit_data[~reddit_data.subreddit.isin(exclude)]
		new_labels = ['control'] * reddit_data_controls.shape[0]
		reddit_data_controls['subreddit'] = new_labels
		# Subsample to match length of reddit_data
		subsample_controls = reddit_data_sr.shape[0]
		reddit_data_controls = subsample_df(reddit_data_controls, subsample_controls)

		# add labels
		reddit_data = pd.concat([reddit_data_sr,reddit_data_controls])

		return reddit_data




