
import numpy as np
import pandas as pd

def subsample_df(df, subsample=False):
	# subsample rows to balance classes
	if type(subsample) == float:
		subsample = int(df.shape[0]*subsample)
	df = df.reset_index(drop=True)
	df2 = df.loc[np.random.choice(df.index,subsample, replace=False)]
	return df2

def multiclass(input_dir, subreddits, pre_or_post = 'pre', subsample=None):
	reddit_data = pd.read_csv(input_dir +subreddits[0]+'_{}_features.csv'.format(pre_or_post), index_col=False)
	# remove author duplicates and shuffle so we dont keep only first posts in time
	reddit_data = reddit_data.sample(frac=1)
	reddit_data = reddit_data.drop_duplicates(subset='author', keep='first')
	reddit_data = reddit_data.reset_index(drop=True)
	print('before:', subreddits[0], reddit_data.shape)
	if subsample:
		reddit_data = subsample_df(reddit_data, subsample=subsample)
		print('after:', subreddits[0], reddit_data.shape)

	for i in np.arange(1, len(subreddits)):
		new_data = pd.read_csv(input_dir+subreddits[i]+'_{}_features.csv'.format(pre_or_post))
		# remove author duplicates and shuffle so we dont keep only first posts in time
		new_data = new_data.sample(frac=1)
		new_data = new_data.drop_duplicates(subset='author', keep='first')
		new_data = new_data.reset_index(drop=True)
		print('before:', subreddits[i], new_data.shape)
		if subsample:
			new_data = subsample_df(new_data, subsample=subsample)
			print('after:',subreddits[i], new_data.shape)

		reddit_data = pd.concat([reddit_data, new_data], axis=0)

	return reddit_data



def binary(input_dir, subreddit, control_subreddits, pre_or_post = 'pre', subsample=None):
	if pre_or_post == 'post':
		# collect mid-pandemic data
		reddit_data = multiclass(input_dir, control_subreddits, pre_or_post=pre_or_post, subsample=subsample)

	if pre_or_post == 'pre':
		# collect pre-pandemic data
		reddit_data = multiclass(input_dir, control_subreddits, pre_or_post=pre_or_post,
		                         subsample=subsample)
		reddit_data_sr = reddit_data[reddit_data.subreddit == subreddit]
		# create control df: remove the treatment and/or overlapping subreddits
		exclude = [subreddit] # Exclude certain the same or overlapping subreddits from control group
		reddit_data_controls = reddit_data[~reddit_data.subreddit.isin(exclude)]
		new_labels = ['control'] * reddit_data_controls.shape[0]
		reddit_data_controls['subreddit'] = new_labels
		print(reddit_data_sr .shape, reddit_data_controls.shape)
		# Subsample to match length of reddit_data
		reddit_data_controls = subsample_df(reddit_data_controls, reddit_data_sr.shape[0])

		# add labels
		reddit_data = pd.concat([reddit_data_sr,reddit_data_controls])

	return reddit_data

# if subreddit == 'healthanxiety':
# 	exclude = ['healthanxiety','anxiety']
# elif subreddit == 'socialanxiety':
# 	exclude = ['socialanxiety', 'anxiety']
# elif subreddit == 'anxiety':
# 	exclude = ['anxiety','healthanxiety','socialanxiety']
# elif subreddit == 'depression':
# 	exclude = ['depression','suicidewatch']
# elif subreddit == 'suicidewatch':
# 	exclude = ['suicidewatch','depression']
# else: