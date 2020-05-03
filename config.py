'''

Author: Daniel M. Low
License: Apache 2.0

'''
import argparse
import ast
import sys

toy = False
run_version_number = 2 #version
run_modelN = 2
run_final_model = False
subsample = 5600  # 10k should be good. False or 0.2
include_subreddits_overN = False #3600 #remove subreddits under this number
subsample_subreddits_overN = False #25000  # Load less, subsample if dataset > N, set to 0 for all
model = 'vector' # 'elastic-net', 'xgboost','extra-trees' 'svm' 'gru' 'lstm' #TODO: the specific version determined by task
dim_reduction = False
cv = 5

input_dir = './../../datum/reddit/input/'
output_dir = './../../datum/reddit/output/'



# Others:
# task = 'multiclass' # 'binary' 'multiclass' 'regression'
# text_features = ['automated_readability_index', 'coleman_liau_index', 'flesch_kincaid_grade_level', 'flesch_reading_ease', 'gulpease_index', 'gunning_fog_index', 'lix', 'smog_index', 'wiener_sachtextformel', 'n_chars', 'n_long_words', 'n_monosyllable_words', 'n_polysyllable_words', 'n_sents', 'n_syllables', 'n_unique_words', 'n_words', 'sent_neg', 'sent_neu', 'sent_pos', 'sent_compound', 'punctuation', '1st_pers', '2nd_pers', '3rd_pers', 'achievement', 'adverbs', 'affective_processes', 'anger', 'anxiety', 'articles_article', 'assent', 'auxiliary_verbs', 'biological', 'body', 'causation', 'certainty', 'cognitive', 'common_verbs', 'conjunctions', 'death', 'discrepancy', 'exclusive', 'family', 'feel', 'fillers', 'friends', 'future_tense', 'health', 'hear', 'home', 'humans', 'impersonal_pronouns', 'inclusive', 'ingestion', 'inhibition', 'insight', 'leisure', 'money', 'motion', 'negations', 'negative_emotion', 'nonfluencies', 'numbers', 'past_tense', 'perceptual_processes', 'personal_pronouns', 'positive_emotion', 'prepositions', 'present_tense', 'quantifiers', 'relativity', 'religion', 'sadness', 'see', 'sexual', 'social_processes', 'space', 'swear_words', 'tentative', 'time', 'total_functional', 'total_pronouns', 'work'] # TODO: add 'tfidf', 'lda', 'words'
# resampling = 'bootstrapping' #'cross_validation' or 'bootstrapping' or False, if False, then evaluate test set.
# predict_proba = False #


def str2boolean(s):
	if s == 'False':
		s_new = False
	else:
		s_new = True
	return s_new

# try:
# this will replace above if defined in command line
# if len(sys.argv) < 2:
#     print("You haven't specified any arguments. Use -h to get more details on how to use this command.")
#     sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('--job_array_task_id',
                    help='default: ${SLURM_ARRAY_TASK_ID} or 1. When using job arrays, this will be set by the bash script by ${SLURM_ARRAY_TASK_ID} or set to 1, which will be substracted below by 1 for zero indexing')
# parser.add_argument('--job_array_start',
#                     help='default: 0. job arrays have a limit of 999 tasks. If you want to run 1200 tasks, first run array=1-999 and use 0 here, then to start 1000, change here to 1000, and array=1-200')
parser.add_argument('--toy', help='run quickly with less labels, parameters and splits')
parser.add_argument('--run_version_number',
                    help='default: 0. if you need to run the same model but output to a different directory, change number')
# parser.add_argument('--modality', help='audio or text or audio_text')
# parser.add_argument('--model', help="'xgboost','extra-trees' 'svm' 'gru' 'lstm' etc")
parser.add_argument('--dim_reduction', help="True or False")
# parser.add_argument('--audio_features', help='text_audio_notTrimmed_compare16_freeresp')
# parser.add_argument('--text_features',
#                     help="['use','tfidf','sentiment', 'words', 'liwc', 'punctuation_count', 'pauses']")
# parser.add_argument('--resampling_inner_loop', help="bootstrapping or cross_validation")

args = parser.parse_args()
toy = str2boolean(args.toy)
run_version_number = int(args.run_version_number)
run_modelN = int(args.job_array_task_id) - 1
dim_reduction = str2boolean(args.dim_reduction)

print('args: ',args)

# except:
# 	print('=========Parser failed======')
# 	pass



# Define subreddits
# subreddits = ['meditation','addiction', 'adhd','anxiety', 'autism', 'bipolarreddit', 'bpd', 'depression', 'healthanxiety', 'ptsd', 'schizophrenia', 'socialanxiety', 'suicidewatch']
subreddits = ['meditation','mindfulness','EDAnonymous','addiction','alcoholism', 'adhd','anxiety', 'autism', 'bipolarreddit', 'bpd', 'depression', 'healthanxiety', 'ptsd', 'schizophrenia', 'socialanxiety', 'suicidewatch']

if run_version_number in [2,3]:
	subreddits = ['alcoholism', 'bipolarreddit', 'depression', 'healthanxiety', 'lonely', 'schizophrenia', 'socialanxiety', 'suicidewatch']
	subsample = 5600 #for alcoholism


if run_version_number in [4]:
	subreddits = ['healthanxiety', 'lonely', 'socialanxiety', 'suicidewatch']
	subsample = 11000  # for alcoholism


# subreddits = ['healthanxiety', 'schizophrenia', 'socialanxiety', 'suicidewatch']
# subreddits.sort()

# subreddits_remove = ['mindfulness', 'meditation', 'alcoholism', 'COVID19_support'] # small ones
# try: [subreddits.remove(n) for n in subreddits_remove]
# except: pass

if toy:
	subsample = 80
	subreddits = subreddits[:2]







# if __name__=='__main__':
# The parser is only called if this script is called as a script/executable (via command line) but not when imported by another script

	# modality = args.modality
	# model = args.model
	# dim_reduction = str2boolean(args.dim_reduction)

	# audio_features = args.audio_features
	# if audio_features == None:
	#     audio_features = 'text_audio_notTrimmed_compare16_freeresp'
	#
	# resampling_inner_loop = args.resampling_inner_loop
	# if resampling_inner_loop == None:
	#     resampling_inner_loop = 'bootstrapping'
	#
	#

	#

	#
	# # Convert the argparse.Namespace to a dictionary: vars(args)
	# main(vars(args))
	# sys.exit(0)

	# python3 run.py --job_array_task_id=1 --job_array_start=0 --toy=False --run_number=2 --modality=audio_text --model=elastic-net --dim_reduction=False --text_features="['use','tfidf','sentiment', 'words', 'liwc', 'punctuation_count', 'pauses']" --resampling_inner_loop='cross_validation'
	# python3 run.py --job_array_task_id=1 --job_array_start=0 --toy=False --run_number=2 --modality=audio_text --model=gru --dim_reduction=False --text_features="['use','sentiment', 'words', 'liwc', 'punctuation_count', 'pauses']" --resampling_inner_loop='cross_validation'

	# toy_subsample = 20 # percent of dataset
	# sample = 'interview' # 'utterance' or 'interview'

	#
	# try: slurm_array = int(sys.argv[2])  # if you need more than 999 slurm array, then change to 1000 here
	# except: slurm_array = 1
	#
